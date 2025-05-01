import os
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

# Directory paths
ALS_MODEL_PATH = "notebooks/als_model"
MOVIE_FEATURES_PATH = "notebooks/movie_features"
USER_PROFILES_PATH = "notebooks/user_profiles"
PROCESSED_DATA_PATH = "notebooks/processed_movie_data"
TRAIN_DATA_PATH = "notebooks/train_data"
TEST_DATA_PATH = "notebooks/test_data"

# Create directories if they don't exist
for path in [ALS_MODEL_PATH, MOVIE_FEATURES_PATH, USER_PROFILES_PATH, 
             PROCESSED_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH]:
    os.makedirs(path, exist_ok=True)

def load_movielens_data(base_path="ml-latest-small"):
    """
    Load MovieLens dataset files
    
    Args:
        base_path: Path to the MovieLens dataset directory
        
    Returns:
        Tuple of (movies_df, ratings_df)
    """
    print(f"Loading MovieLens data from {base_path}...")
    
    # Load movies data
    movies_file = os.path.join(base_path, "movies.csv")
    movies_df = pd.read_csv(movies_file)
    print(f"Loaded {len(movies_df)} movies")
    
    # Load ratings data
    ratings_file = os.path.join(base_path, "ratings.csv")
    ratings_df = pd.read_csv(ratings_file)
    print(f"Loaded {len(ratings_df)} ratings from {ratings_df['userId'].nunique()} users")
    
    return movies_df, ratings_df


def extract_movie_features(movies_df):
    """
    Extract content features from movie data
    
    Args:
        movies_df: DataFrame containing movie data
        
    Returns:
        Dictionary mapping movie IDs to feature vectors
    """
    print("Extracting movie features...")
    
    # Preprocess movie titles and genres
    movies_df['clean_title'] = movies_df['title'].str.replace(r'\(\d{4}\)', '', regex=True)
    movies_df['clean_title'] = movies_df['clean_title'].str.strip()
    
    # Combine title and genres for feature extraction
    movies_df['content'] = movies_df['clean_title'] + " " + movies_df['genres'].str.replace('|', ' ')
    
    # Use TF-IDF to extract features
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies_df['content'])
    
    # Reduce dimensions with SVD for better performance
    svd = TruncatedSVD(n_components=100, random_state=42)
    movie_features_matrix = svd.fit_transform(tfidf_matrix)
    
    # Create a dictionary mapping movie IDs to feature vectors
    movie_features = {}
    for i, movie_id in enumerate(movies_df['movieId']):
        movie_features[movie_id] = movie_features_matrix[i].tolist()
    
    print(f"Extracted features for {len(movie_features)} movies (vector size: {len(next(iter(movie_features.values())))})")
    
    return movie_features


def split_train_test_data(ratings_df, test_ratio=0.2, random_state=42):
    """
    Split ratings data into training and testing sets
    
    Args:
        ratings_df: DataFrame containing ratings
        test_ratio: Ratio of test data (default: 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    print(f"Splitting data into train ({1-test_ratio:.0%}) and test ({test_ratio:.0%}) sets...")
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Create a mask for the test set
    mask = np.random.rand(len(ratings_df)) < test_ratio
    
    # Split the data
    test_df = ratings_df[mask]
    train_df = ratings_df[~mask]
    
    print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
    
    return train_df, test_df


def train_als_model(spark, train_df, test_df=None, 
                   rank=10, max_iter=10, reg_param=0.01):
    """
    Train ALS model for collaborative filtering
    
    Args:
        spark: SparkSession instance
        train_df: DataFrame containing training data
        test_df: DataFrame containing test data (for evaluation)
        rank: Number of latent factors
        max_iter: Maximum number of iterations
        reg_param: Regularization parameter
        
    Returns:
        Trained ALS model
    """
    print(f"Training ALS model (rank={rank}, max_iter={max_iter}, reg_param={reg_param})...")
    
    # Define the schema for the ratings data
    schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", FloatType(), True)
    ])
    
    # Convert pandas DataFrames to Spark DataFrames
    train_data = [(int(row.userId), int(row.movieId), float(row.rating)) for _, row in train_df.iterrows()]
    train_spark = spark.createDataFrame(train_data, schema=schema)
    
    # Create and train the ALS model
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        coldStartStrategy="drop",
        nonnegative=True
    )
    
    model = als.fit(train_spark)
    
    # Evaluate the model on test data if provided
    if test_df is not None:
        test_data = [(int(row.userId), int(row.movieId), float(row.rating)) for _, row in test_df.iterrows()]
        test_spark = spark.createDataFrame(test_data, schema=schema)
        
        # Make predictions
        predictions = model.transform(test_spark)
        predictions = predictions.filter(predictions.prediction.isNotNull())
        
        # Evaluate using RMSE
        evaluator = RegressionEvaluator(
            metricName="rmse", 
            labelCol="rating", 
            predictionCol="prediction"
        )
        rmse = evaluator.evaluate(predictions)
        print(f"Root-mean-square error on test data: {rmse:.4f}")
    
    return model


def process_and_save_data():
    """
    Process MovieLens data, extract features, train models, and save everything
    """
    print("Starting data processing pipeline...")
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("MovieLensDataPrep") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    try:
        # 1. Load data
        movies_df, ratings_df = load_movielens_data()
        
        # 2. Extract movie features
        movie_features = extract_movie_features(movies_df)
        
        # 3. Split data into train and test sets
        train_df, test_df = split_train_test_data(ratings_df)
        
        # 4. Save processed data
        print(f"Saving processed movie data to {PROCESSED_DATA_PATH}...")
        movies_df.to_csv(f"{PROCESSED_DATA_PATH}/movies.csv", index=False)
        ratings_df.to_csv(f"{PROCESSED_DATA_PATH}/ratings.csv", index=False)
        
        # 5. Save train and test data
        print(f"Saving train/test data...")
        train_df.to_csv(f"{TRAIN_DATA_PATH}/ratings.csv", index=False)
        test_df.to_csv(f"{TEST_DATA_PATH}/ratings.csv", index=False)
        
        # 6. Save movie features
        print(f"Saving movie features to {MOVIE_FEATURES_PATH}...")
        with open(f"{MOVIE_FEATURES_PATH}/features.pkl", "wb") as f:
            pickle.dump(movie_features, f)
        
        # 7. Train and save ALS model
        print(f"Training ALS model...")
        als_model = train_als_model(spark, train_df, test_df)
        
        print(f"Saving ALS model to {ALS_MODEL_PATH}...")
        als_model.write().overwrite().save(ALS_MODEL_PATH)
        
        print("Data processing pipeline completed successfully!")
        
    finally:
        # Stop Spark session
        spark.stop()


def test_recommender():
    """
    Test the hybrid recommender system
    """
    print("\nTesting hybrid recommender system...")
    
    # Import here to avoid circular imports
    from content_based import HybridRecommender
    
    # Initialize the recommender
    recommender = HybridRecommender(content_weight=0.3)
    recommender.initialize()
    
    # Test collaborative filtering
    print("\nTesting collaborative filtering recommendations:")
    user_id = 1  # Choose a user ID that exists in the dataset
    cf_recs = recommender._get_als_recommendations(user_id, n=5)
    for i, rec in enumerate(cf_recs, 1):
        print(f"{i}. {rec['title']} (Score: {rec['score']:.2f})")
    
    # Test content-based filtering
    print("\nTesting content-based recommendations:")
    if user_id in recommender.user_profiles:
        cb_recs = recommender._get_content_recommendations(user_id, n=5)
        for i, rec in enumerate(cb_recs, 1):
            print(f"{i}. {rec['title']} (Score: {rec['score']:.2f})")
    else:
        print(f"No user profile for user {user_id}. Try with a different user ID.")
    
    # Test hybrid recommendations
    print("\nTesting hybrid recommendations:")
    hybrid_recs = recommender.get_recommendations(user_id, n=5)
    for i, rec in enumerate(hybrid_recs, 1):
        print(f"{i}. {rec['title']} (Score: {rec['score']:.2f}, Type: {rec['type']})")
    
    # Test similar movies
    print("\nTesting similar movies:")
    movie_id = 1  # Toy Story
    similar_movies = recommender.movie_features.get_similar_movies(movie_id, n=5)
    for i, movie in enumerate(similar_movies, 1):
        print(f"{i}. {movie['title']} (Similarity: {movie['similarity']:.2f})")


if __name__ == "__main__":
    # Process and save data
    process_and_save_data()
    
    # Test the recommender system
    test_recommender()
