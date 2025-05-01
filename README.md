# Hybrid Movie Recommendation System

A scalable and efficient movie recommendation system built with Apache Spark that combines collaborative filtering and content-based approaches to provide personalized movie recommendations.

## Project Overview

This project implements a hybrid recommendation system that leverages both the power of collaborative filtering (what similar users liked) and content-based filtering (movies with similar attributes) to provide high-quality movie recommendations. The system uses the MovieLens dataset and Apache Spark's MLlib to handle large-scale data processing and machine learning.

Key features:
- Collaborative filtering using Alternating Least Squares (ALS)
- Content-based filtering using TF-IDF and cosine similarity
- Hybrid approach with configurable weights to balance both methods
- Web interface built with Flask for easy interaction
- Scalable architecture suitable for large datasets

## Project Structure

```
bdi-project/
├── app.py                     # Flask web application
├── run_recommender.py         # Command-line demo script
├── ml-latest-small/           # MovieLens dataset
├── movie_rec_env/             # Virtual environment
├── notebooks/
│   ├── content_based.py       # Hybrid recommender implementation
│   ├── prepare_data.py        # Data processing and model training
│   ├── als_model/             # Saved ALS model
│   ├── movie_features/        # Extracted movie features
│   ├── user_profiles/         # User preference profiles
│   ├── processed_movie_data/  # Processed dataset
│   ├── train_data/            # Training data split
│   └── test_data/             # Testing data split
└── README.md                  # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Java 8 or higher (required for Apache Spark)
- MovieLens dataset (included in the repository)

### Setup

1. Clone the repository:
   ```
   git clone (https://github.com/Tushar4059x/movie_recommendation_system)
   cd bdi-project
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv movie_rec_env
   source movie_rec_env/bin/activate  # On Windows: movie_rec_env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   If no requirements.txt exists, install the following packages:
   ```
   pip install pandas numpy scikit-learn flask pyspark
   ```

## Usage

### Data Preparation and Model Training

Before running the recommender, you need to process the data and train the models. This step is automatically handled when you run the provided scripts, but you can also run it separately:

```
python notebooks/prepare_data.py
```

This script:
1. Loads the MovieLens dataset
2. Extracts features from movie data (TF-IDF on genres and titles)
3. Trains the ALS collaborative filtering model
4. Creates and saves user profiles

### Command-line Interface

Run the demonstration script to see the recommender in action:

```
python run_recommender.py
```

This will:
1. Check if all necessary models and files exist
2. Run data preparation if needed
3. Demonstrate various recommendation capabilities
4. Show examples of how the system works

### Web Interface

Start the web application:

```
python app.py
```

Then open your browser and navigate to: http://localhost:5000/

The web app provides the following endpoints:
- `/recommend` - Get recommendations for a user
- `/movie/<movie_id>/similar` - Find similar movies
- `/user/<user_id>/profile` - View a user's ratings
- `/movies` - List all movies
- `/stats` - View dataset statistics

## Implementation Details

### Hybrid Recommendation Approach

The system combines two recommendation strategies:

1. **Collaborative Filtering (CF)**: Uses Apache Spark's Alternating Least Squares (ALS) algorithm to learn latent factors that represent user preferences and movie characteristics. This approach recommends movies based on what similar users have enjoyed.

2. **Content-Based Filtering (CB)**: Extracts features from movie metadata (titles and genres) using TF-IDF and dimensionality reduction. These features are used to calculate similarity between movies and to build user profiles based on their ratings.

3. **Hybrid Model**: Combines recommendations from both approaches using a configurable weight parameter (`content_weight`). This allows for a personalized balance between the two methods:
   - Setting `content_weight=0.0` uses pure collaborative filtering
   - Setting `content_weight=1.0` uses pure content-based filtering
   - Values in between blend the two approaches

The hybrid approach helps mitigate common issues like the cold-start problem (for new users or movies) and provides more diverse and relevant recommendations.

## Technologies Used

- **Apache Spark**: For large-scale data processing and the ALS algorithm
- **PySpark MLlib**: For implementing collaborative filtering
- **scikit-learn**: For TF-IDF feature extraction and similarity calculations
- **Flask**: For the web application interface
- **Pandas & NumPy**: For data manipulation and analysis
- **MovieLens Dataset**: For movie ratings data

## License

This project uses the MovieLens dataset, which is provided by GroupLens Research and subject to their terms of use.

## Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for providing the dataset
- [Apache Spark](https://spark.apache.org/) for the distributed computing framework

