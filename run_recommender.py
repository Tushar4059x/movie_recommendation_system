#!/usr/bin/env python
"""
Movie Recommendation System Demo

This script demonstrates the hybrid movie recommendation system's capabilities.
It checks if necessary models and data files exist, prepares them if needed,
and shows various examples of the recommendation system in action.
"""

import os
import sys
import subprocess
import time
import pandas as pd
from notebooks.content_based import HybridRecommender

# ANSI color codes for terminal output formatting
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print a formatted header text"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")

def print_section(text):
    """Print a formatted section text"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * len(text)}{Colors.ENDC}")

def check_requirements():
    """Check if all necessary files and models exist"""
    print_section("Checking if all necessary files and models exist...")
    
    required_paths = [
        "notebooks/als_model",
        "notebooks/movie_features/features.pkl",
        "notebooks/user_profiles/profiles.pkl",
        "notebooks/processed_movie_data/movies.csv",
        "notebooks/processed_movie_data/ratings.csv"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
            print(f"{Colors.YELLOW}Missing: {path}{Colors.ENDC}")
    
    if missing_paths:
        print(f"\n{Colors.YELLOW}Some required files are missing. Need to run data preparation.{Colors.ENDC}")
        return False
    else:
        print(f"{Colors.GREEN}All required files and models exist!{Colors.ENDC}")
        return True

def prepare_data():
    """Run the data preparation script"""
    print_section("Preparing data and training models...")
    print(f"{Colors.YELLOW}This may take some time depending on your hardware...{Colors.ENDC}")
    
    start_time = time.time()
    
    try:
        # Run the prepare_data.py script using subprocess
        result = subprocess.run(
            [sys.executable, "notebooks/prepare_data.py"],
            check=True,
            text=True
        )
        
        if result.returncode == 0:
            elapsed_time = time.time() - start_time
            print(f"{Colors.GREEN}Data preparation completed successfully in {elapsed_time:.2f} seconds!{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.RED}Data preparation failed with exit code {result.returncode}{Colors.ENDC}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Error during data preparation: {e}{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Unexpected error: {e}{Colors.ENDC}")
        return False

def get_user_recommendations(recommender, user_id, n=5):
    """Get recommendations for a specific user"""
    recommendations = recommender.get_recommendations(user_id, n=n)
    
    print(f"Top {n} recommendations for User {user_id}:")
    if not recommendations:
        print(f"{Colors.YELLOW}No recommendations found for this user.{Colors.ENDC}")
        return
    
    for i, rec in enumerate(recommendations, 1):
        # Format the output with color based on recommendation type
        type_color = Colors.GREEN if rec['type'] == 'collaborative' else Colors.BLUE
        print(f"{i}. {rec['title']} {Colors.BOLD}({rec['genres']}){Colors.ENDC}")
        print(f"   Score: {rec['score']:.2f} | Type: {type_color}{rec['type']}{Colors.ENDC}")

def get_similar_movies(recommender, movie_id, movie_title, n=5):
    """Get movies similar to a specific movie"""
    similar_movies = recommender.movie_features.get_similar_movies(movie_id, n=n)
    
    print(f"Top {n} movies similar to '{movie_title}':")
    if not similar_movies:
        print(f"{Colors.YELLOW}No similar movies found.{Colors.ENDC}")
        return
    
    for i, movie in enumerate(similar_movies, 1):
        print(f"{i}. {movie['title']} {Colors.BOLD}({movie['genres']}){Colors.ENDC}")
        print(f"   Similarity: {movie['similarity']:.2f}")

def compare_recommendation_weights(recommender, user_id, n=5):
    """Compare recommendations with different content weights"""
    weights = [0.0, 0.3, 0.7, 1.0]
    
    print(f"Comparing recommendations for User {user_id} with different content weights:")
    print(f"  0.0 = Pure collaborative filtering")
    print(f"  1.0 = Pure content-based filtering")
    
    for weight in weights:
        print(f"\n{Colors.BOLD}Content Weight: {weight}{Colors.ENDC}")
        # Create a temporary recommender with this weight
        temp_recommender = HybridRecommender(content_weight=weight)
        temp_recommender.spark = recommender.spark
        temp_recommender.als_model = recommender.als_model
        temp_recommender.movie_features = recommender.movie_features
        temp_recommender.user_profiles = recommender.user_profiles
        temp_recommender.movies_df = recommender.movies_df
        temp_recommender.ratings_df = recommender.ratings_df
        
        recommendations = temp_recommender.get_recommendations(user_id, n=n)
        
        if not recommendations:
            print(f"{Colors.YELLOW}No recommendations found.{Colors.ENDC}")
            continue
        
        for i, rec in enumerate(recommendations, 1):
            type_color = Colors.GREEN if rec['type'] == 'collaborative' else Colors.BLUE
            print(f"{i}. {rec['title']} {Colors.BOLD}({rec['genres']}){Colors.ENDC}")
            print(f"   Score: {rec['score']:.2f} | Type: {type_color}{rec['type']}{Colors.ENDC}")

def get_top_movies(movies_df, ratings_df, n=5):
    """Get top rated movies (with at least 100 ratings)"""
    # Calculate average ratings and count
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['count', 'mean']
    })
    movie_stats.columns = ['count', 'average']
    
    # Filter movies with at least 100 ratings
    popular_movies = movie_stats[movie_stats['count'] >= 100]
    
    # Sort by average rating
    top_movies = popular_movies.sort_values('average', ascending=False).head(n)
    
    # Merge with movie details
    top_movies_df = pd.merge(
        top_movies.reset_index(), 
        movies_df, 
        on='movieId'
    )
    
    return top_movies_df

def main():
    """Main function to demonstrate the recommendation system"""
    print_header("Movie Recommendation System Demo")
    
    # Check if required files exist
    if not check_requirements():
        # Run data preparation if files are missing
        if not prepare_data():
            print(f"{Colors.RED}Failed to prepare data. Exiting.{Colors.ENDC}")
            return
    
    print_section("Initializing recommendation system...")
    # Initialize the recommender
    recommender = HybridRecommender(content_weight=0.3)
    recommender.initialize()
    print(f"{Colors.GREEN}Recommendation system initialized successfully!{Colors.ENDC}")
    
    # Load movies and ratings for reference
    movies_df = pd.read_csv("ml-latest-small/movies.csv")
    ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
    
    # Print some dataset statistics
    print_section("Dataset Statistics")
    print(f"Total movies: {len(movies_df)}")
    print(f"Total users: {ratings_df['userId'].nunique()}")
    print(f"Total ratings: {len(ratings_df)}")
    print(f"Rating scale: {ratings_df['rating'].min()} to {ratings_df['rating'].max()}")
    
    # Get top movies
    print_section("Top Rated Movies (with at least 100 ratings)")
    top_movies = get_top_movies(movies_df, ratings_df)
    for i, (_, row) in enumerate(top_movies.iterrows(), 1):
        print(f"{i}. {row['title']} {Colors.BOLD}({row['genres']}){Colors.ENDC}")
        print(f"   Average Rating: {row['average']:.2f} from {int(row['count'])} ratings")
    
    # Examples of getting recommendations for different users
    print_header("Recommendation Examples")
    
    # Example 1: Get recommendations for an active user
    active_users = ratings_df['userId'].value_counts().head(3).index.tolist()
    print_section(f"Example 1: Recommendations for active users")
    for user_id in active_users:
        user_ratings_count = ratings_df[ratings_df['userId'] == user_id].shape[0]
        print(f"\n{Colors.BOLD}User {user_id} ({user_ratings_count} ratings){Colors.ENDC}")
        get_user_recommendations(recommender, user_id)
    
    # Example 2: Get similar movies
    print_section(f"Example 2: Finding similar movies")
    # Find a popular movie to use as an example
    sample_movie_id = 1   # Toy Story
    sample_movie_title = movies_df[movies_df['movieId'] == sample_movie_id]['title'].iloc[0]
    get_similar_movies(recommender, sample_movie_id, sample_movie_title)
    
    # Another example with a different movie
    sample_movie_id = 356   # Forrest Gump
    sample_movie_title = movies_df[movies_df['movieId'] == sample_movie_id]['title'].iloc[0]
    get_similar_movies(recommender, sample_movie_id, sample_movie_title)
    
    # Example 3: Compare different content weights
    print_section(f"Example 3: Comparing different content weights")
    user_id = active_users[0]  # Use the first active user
    compare_recommendation_weights(recommender, user_id)
    
    # Example 4: Using with the web application
    print_header("Using the Web Application")
    print(f"To start the web application, run the following command in your terminal:")
    print(f"\n{Colors.BOLD}python app.py{Colors.ENDC}\n")
    print(f"Then, open a web browser and navigate to: {Colors.BOLD}http://localhost:5000/{Colors.ENDC}")
    print(f"\nThe web app provides the following endpoints:")
    print(f"- {Colors.BOLD}/recommend{Colors.ENDC} - Get recommendations for a user")
    print(f"- {Colors.BOLD}/movie/<movie_id>/similar{Colors.ENDC} - Find similar movies")
    print(f"- {Colors.BOLD}/user/<user_id>/profile{Colors.ENDC} - View a user's ratings")
    print(f"- {Colors.BOLD}/movies{Colors.ENDC} - List all movies")
    print(f"- {Colors.BOLD}/stats{Colors.ENDC} - View dataset statistics")
    
    # Clean up
    recommender.spark.stop()
    print(f"\n{Colors.GREEN}Demo completed successfully!{Colors.ENDC}")

if __name__ == "__main__":
    main()

