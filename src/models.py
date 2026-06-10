import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNBasic, SVD

def train_user_based_cf(df, k=40):
    """Train a User-Based Collaborative Filtering model using Surprise KNNBasic."""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    sim_options = {
        'name': 'cosine',
        'user_based': True
    }
    algo = KNNBasic(k=k, sim_options=sim_options, random_state=42, verbose=False)
    algo.fit(trainset)
    return algo

def train_item_based_cf(df, k=40):
    """Train an Item-Based Collaborative Filtering model using Surprise KNNBasic."""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    sim_options = {
        'name': 'cosine',
        'user_based': False
    }
    algo = KNNBasic(k=k, sim_options=sim_options, random_state=42, verbose=False)
    algo.fit(trainset)
    return algo

def train_svd(df):
    """Train a Matrix Factorization (SVD) model using Surprise SVD."""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    algo = SVD(random_state=42)
    algo.fit(trainset)
    return algo

def get_cf_recommendations(algo, df, user_id, top_n=10):
    """Generate recommendations using a Surprise trained algorithm for a user."""
    # Find all unique movies
    all_movies = df[['movie_id', 'title']].drop_duplicates(subset=['movie_id'])
    
    # Find movies already rated by the user
    rated_movie_ids = set(df[df['user_id'] == user_id]['movie_id'].unique())
    
    # Predict rating for each unrated movie
    recommendations = []
    for _, row in all_movies.iterrows():
        mid = int(row['movie_id'])
        title = row['title']
        if mid not in rated_movie_ids:
            pred = algo.predict(uid=user_id, iid=mid)
            recommendations.append({
                'movie_id': mid,
                'title': title,
                'estimated_rating': float(pred.est)
            })
            
    # Sort recommendations by estimated rating descending
    recommendations.sort(key=lambda x: x['estimated_rating'], reverse=True)
    return recommendations[:top_n]

def get_content_based_recommendations(df, user_id, top_n=10):
    """Generate content-based recommendations for a user based on movie genres and TF-IDF."""
    # Get unique movies
    df_movies = df[['movie_id', 'title', 'genres']].drop_duplicates(subset=['movie_id']).copy()
    df_movies = df_movies.sort_values('movie_id').reset_index(drop=True)
    
    # Clean genres string (replace pipe with space)
    df_movies['genre_clean'] = df_movies['genres'].str.replace('|', ' ', regex=False).fillna('')
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = vectorizer.fit_transform(df_movies['genre_clean'])
    
    # Map movie_id to index in tfidf_matrix
    movie_id_to_idx = {row['movie_id']: idx for idx, row in df_movies.iterrows()}
    
    # Get user history
    user_ratings = df[df['user_id'] == user_id]
    if user_ratings.empty:
        # If user has no history, return empty or handle elsewhere (cold-start)
        return []
        
    user_movie_ids = user_ratings['movie_id'].values
    user_movie_ratings = user_ratings['rating'].values
    
    # Compute user profile vector (ratings-weighted average of TF-IDF vectors)
    user_profile = np.zeros(tfidf_matrix.shape[1])
    for mid, rating in zip(user_movie_ids, user_movie_ratings):
        if mid in movie_id_to_idx:
            idx = movie_id_to_idx[mid]
            user_profile += rating * tfidf_matrix[idx].toarray()[0]
            
    # Normalize profile vector
    norm = np.linalg.norm(user_profile)
    if norm > 0:
        user_profile = user_profile / norm
        
    # Compute cosine similarity between user profile and all movies
    sim_scores = cosine_similarity(user_profile.reshape(1, -1), tfidf_matrix).flatten()
    df_movies['similarity_score'] = sim_scores
    
    # Filter out movies user has already rated
    df_unrated = df_movies[~df_movies['movie_id'].isin(user_movie_ids)]
    
    # Sort and return top N
    df_rec = df_unrated.sort_values(by='similarity_score', ascending=False).head(top_n)
    
    recommendations = []
    for _, row in df_rec.iterrows():
        recommendations.append({
            'movie_id': int(row['movie_id']),
            'title': row['title'],
            'similarity_score': float(row['similarity_score'])
        })
        
    return recommendations

def get_cold_start_recommendations(df, min_ratings=50, top_n=10):
    """Generate cold start recommendations based on the most popular movies (highest average rating)."""
    # Group by movie to calculate mean rating and rating count
    movie_stats = df.groupby('movie_id').agg(
        average_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()
    
    # Get movie titles
    df_titles = df[['movie_id', 'title']].drop_duplicates(subset=['movie_id'])
    movie_stats = pd.merge(movie_stats, df_titles, on='movie_id')
    
    # Filter by minimum ratings to avoid rating bias of single-user ratings
    # If the dataset is tiny or we want to be safe, check if we have enough movies with min_ratings
    popular_movies = movie_stats[movie_stats['rating_count'] >= min_ratings]
    
    # Fallback to lower rating threshold if not enough movies
    if len(popular_movies) < top_n:
        popular_movies = movie_stats
        
    # Sort by average_rating descending, then rating_count descending to break ties
    sorted_movies = popular_movies.sort_values(by=['average_rating', 'rating_count'], ascending=[False, False])
    
    recommendations = []
    for _, row in sorted_movies.head(top_n).iterrows():
        recommendations.append({
            'movie_id': int(row['movie_id']),
            'title': row['title'],
            'average_rating': float(row['average_rating'])
        })
        
    return recommendations
