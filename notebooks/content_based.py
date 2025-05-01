import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class MovieFeatures:
    """Class for handling movie content features and similarity calculations"""
    
    def __init__(self):
        self.movies_df = None
        self.movie_features = None
        self.feature_matrix = None
        self.movie_ids = None
        
    def load(self, movies_path="ml-latest-small/movies.csv", movie_features_path="notebooks/movie_features"):
        """Load movie data and precomputed features"""
        self.movies_df = pd.read_csv(movies_path)
        
        # Load movie features from disk
        if os.path.exists(f"{movie_features_path}/features.pkl"):
            with open(f"{movie_features_path}/features.pkl", "rb") as f:
                self.movie_features = pickle.load(f)
                
            # Extract feature matrix for similarity calculations
            movie_ids = list(self.movie_features.keys())
            self.feature_matrix = np.array([self.movie_features[movie_id] for movie_id in movie_ids])
            self.movie_ids = movie_ids
        else:
            raise FileNotFoundError(f"Movie features not found at {movie_features_path}/features.pkl")
    
    def get_similar_movies(self, movie_id, n=5):
        """Find movies similar to the given movie_id based on content features"""
        if self.movie_features is None:
            raise ValueError("Movie features not loaded. Call load() first.")
            
        if movie_id not in self.movie_features:
            return []
        
        # Get feature vector for the target movie
        movie_idx = self.movie_ids.index(movie_id)
        movie_vector = self.feature_matrix[movie_idx].reshape(1, -1)
        
        # Calculate cosine similarity with all movies
        similarity_scores = cosine_similarity(movie_vector, self.feature_matrix).flatten()
        
        # Get indices of top similar movies (excluding the movie itself)
        similar_indices = np.argsort(similarity_scores)[::-1][1:n+1]
        
        # Map indices to movie IDs and get movie details
        similar_movies = []
        for idx in similar_indices:
            similar_movie_id = self.movie_ids[idx]
            movie_info = self.movies_df[self.movies_df['movieId'] == similar_movie_id]
            if not movie_info.empty:
                similar_movies.append({
                    'movie_id': int(similar_movie_id),
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres'],
                    'similarity': float(similarity_scores[idx])
                })
        
        return similar_movies


class HybridRecommender:
    """Hybrid recommendation system combining collaborative filtering (ALS) and content-based filtering"""
    
    def __init__(self, content_weight=0.3):
        """
        Initialize the hybrid recommender
        
        Args:
            content_weight: Weight for content-based recommendations (0 to 1)
                            where 0 is pure collaborative and 1 is pure content-based
        """
        self.content_weight = content_weight
        self.spark = None
        self.als_model = None
        self.movie_features = MovieFeatures()
        self.user_profiles = {}
        self.movies_df = None
        self.ratings_df = None
    
    def initialize(self):
        """Initialize the recommender by loading necessary data and models"""
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("HybridMovieRecommender") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        
        # Load movie and rating data
        self.movies_df = pd.read_csv("ml-latest-small/movies.csv")
        self.ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
        
        # Load ALS model
        self.als_model = ALSModel.load("notebooks/als_model")
        
        # Load movie features for content-based filtering
        self.movie_features.load()
        
        # Load or create user profiles for content-based recommendations
        self._load_user_profiles()
    
    def _load_user_profiles(self, user_profiles_path="notebooks/user_profiles"):
        """Load precomputed user profiles or create them from ratings data"""
        if os.path.exists(f"{user_profiles_path}/profiles.pkl"):
            with open(f"{user_profiles_path}/profiles.pkl", "rb") as f:
                self.user_profiles = pickle.load(f)
        else:
            # If no precomputed profiles exist, create them from ratings
            self._create_user_profiles()
    
    def _create_user_profiles(self):
        """Create user profiles based on their ratings and movie features"""
        # Group ratings by user
        user_ratings = self.ratings_df.groupby('userId')
        
        for user_id, ratings in user_ratings:
            # Get the movies this user has rated
            rated_movies = ratings.merge(self.movies_df, on='movieId')
            
            # Skip users with insufficient ratings
            if len(rated_movies) < 5:
                continue
            
            # Calculate weighted average of movie features based on user ratings
            user_profile = np.zeros(len(next(iter(self.movie_features.movie_features.values()))))
            total_weight = 0
            
            for _, row in rated_movies.iterrows():
                movie_id = row['movieId']
                rating = row['rating']
                
                if movie_id in self.movie_features.movie_features:
                    # Use rating as weight (normalize to 0-1 scale)
                    weight = (rating - 2.5) / 2.5  # Center around 0
                    if weight > 0:  # Only consider positive preferences
                        user_profile += weight * np.array(self.movie_features.movie_features[movie_id])
                        total_weight += weight
            
            if total_weight > 0:
                user_profile /= total_weight
                self.user_profiles[user_id] = user_profile
        
        # Save user profiles for future use
        os.makedirs("notebooks/user_profiles", exist_ok=True)
        with open("notebooks/user_profiles/profiles.pkl", "wb") as f:
            pickle.dump(self.user_profiles, f)
    
    def _get_als_recommendations(self, user_id, n=10):
        """Get collaborative filtering recommendations using ALS model"""
        try:
            # Create a single item dataframe with the user ID
            user_df = self.spark.createDataFrame([(user_id,)], ["userId"])
            
            # Generate recommendations
            recommendations = self.als_model.recommendForUserSubset(user_df, n)
            
            if recommendations.count() == 0:
                return []
            
            # Extract recommendation details
            als_recs = recommendations.collect()[0].recommendations
            
            # Convert to list of dictionaries
            result = []
            for rec in als_recs:
                movie_id = rec.movieId
                movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
                if not movie_info.empty:
                    result.append({
                        'movie_id': int(movie_id),
                        'title': movie_info.iloc[0]['title'],
                        'genres': movie_info.iloc[0]['genres'],
                        'score': float(rec.rating),
                        'type': 'collaborative'
                    })
            
            return result
        except Exception as e:
            print(f"Error in ALS recommendations: {e}")
            return []
    
    def _get_content_recommendations(self, user_id, n=10):
        """Get content-based recommendations using user profile"""
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # Calculate similarity between user profile and all movie features
        similarities = []
        for movie_id, features in self.movie_features.movie_features.items():
            # Skip movies the user has already rated
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            if movie_id in user_ratings['movieId'].values:
                continue
            
            similarity = cosine_similarity([user_profile], [features])[0][0]
            similarities.append((movie_id, similarity))
        
        # Sort by similarity and get top n
        top_movies = sorted(similarities, key=lambda x: x[1], reverse=True)[:n]
        
        # Get movie details
        result = []
        for movie_id, score in top_movies:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                result.append({
                    'movie_id': int(movie_id),
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres'],
                    'score': float(score),
                    'type': 'content'
                })
        
        return result
    
    def get_recommendations(self, user_id, n=10):
        """
        Get hybrid recommendations for a user
        
        Args:
            user_id: User ID to get recommendations for
            n: Number of recommendations to return
            
        Returns:
            List of recommended movies with details
        """
        # Get collaborative filtering recommendations
        cf_recs = self._get_als_recommendations(user_id, n=n*2)  # Get more to allow for hybrid mixing
        
        # Get content-based recommendations
        cb_recs = self._get_content_recommendations(user_id, n=n*2)
        
        # If either method fails, return results from the other
        if not cf_recs:
            return cb_recs[:n]
        if not cb_recs:
            return cf_recs[:n]
        
        # Combine recommendations based on content_weight
        # Higher weight means more content-based recommendations
        n_content = int(n * self.content_weight)
        n_collab = n - n_content
        
        # Ensure we have enough recommendations of each type
        n_content = min(n_content, len(cb_recs))
        n_collab = min(n_collab, len(cf_recs))
        
        # Adjust counts if either list is too short
        if n_content < int(n * self.content_weight):
            n_collab = min(n - n_content, len(cf_recs))
        if n_collab < n - int(n * self.content_weight):
            n_content = min(n - n_collab, len(cb_recs))
        
        # Get the top recommendations of each type
        hybrid_recs = cf_recs[:n_collab] + cb_recs[:n_content]
        
        # Sort by score (normalized within each type)
        for rec in hybrid_recs:
            if rec['type'] == 'collaborative':
                # Normalize collaborative scores typically from 0-5 to 0-1
                rec['score'] = rec['score'] / 5.0
        
        # Sort by score and limit to n
        hybrid_recs = sorted(hybrid_recs, key=lambda x: x['score'], reverse=True)[:n]
        
        return hybrid_recs

