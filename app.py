from flask import Flask, render_template, request, jsonify
import pandas as pd
from notebooks.content_based import HybridRecommender

app = Flask(__name__)

# Initialize the recommender system
recommender = HybridRecommender(content_weight=0.3)
recommender.initialize()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.form.get('user_id', type=int)
    if not user_id:
        return jsonify({'error': 'Please provide a valid user ID'})
    
    try:
        recommendations = recommender.get_recommendations(user_id, n=10)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/movie/<int:movie_id>/similar')
def similar_movies(movie_id):
    try:
        similar_movies = recommender.movie_features.get_similar_movies(movie_id, n=5)
        return jsonify({'similar_movies': similar_movies})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/movies')
def get_movies():
    # Return a list of all movies for the UI
    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    movies = movies_df.to_dict('records')
    return jsonify({'movies': movies})

@app.route('/user/<int:user_id>/profile')
def user_profile(user_id):
    try:
        # Get user's rated movies
        ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
        movies_df = pd.read_csv('ml-latest-small/movies.csv')
        
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        if user_ratings.empty:
            return jsonify({'error': 'User not found or has no ratings'})
        
        # Join with movies to get titles
        user_movies = user_ratings.merge(movies_df, on='movieId')
        
        # Convert to list of dictionaries
        rated_movies = []
        for _, row in user_movies.iterrows():
            rated_movies.append({
                'movie_id': int(row['movieId']),
                'title': row['title'],
                'genres': row['genres'],
                'rating': float(row['rating']),
                'timestamp': int(row['timestamp'])
            })
        
        # Sort by rating (highest first)
        rated_movies = sorted(rated_movies, key=lambda x: x['rating'], reverse=True)
        
        return jsonify({
            'user_id': user_id,
            'num_ratings': len(rated_movies),
            'rated_movies': rated_movies
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/stats')
def stats():
    try:
        # Load data
        ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
        movies_df = pd.read_csv('ml-latest-small/movies.csv')
        
        # Calculate simple statistics
        num_users = ratings_df['userId'].nunique()
        num_movies = movies_df.shape[0]
        num_ratings = ratings_df.shape[0]
        avg_rating = ratings_df['rating'].mean()
        
        # Get top-rated movies (with at least 10 ratings)
        movie_ratings = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        })
        movie_ratings.columns = ['count', 'mean']
        popular_movies = movie_ratings[movie_ratings['count'] >= 10].sort_values('mean', ascending=False)
        
        # Get top 5 popular movies
        top_movies = []
        for movie_id, row in popular_movies.head(5).iterrows():
            movie_info = movies_df[movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                top_movies.append({
                    'movie_id': int(movie_id),
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres'],
                    'avg_rating': float(row['mean']),
                    'num_ratings': int(row['count'])
                })
        
        return jsonify({
            'stats': {
                'num_users': num_users,
                'num_movies': num_movies,
                'num_ratings': num_ratings,
                'avg_rating': float(avg_rating)
            },
            'top_movies': top_movies
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
