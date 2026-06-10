import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Movie Recommendation API")

# Global variables to store loaded model and data
model = None
df_data = None
all_movies = None
cold_start_recs = []
user_rated_movies = {}

class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    estimated_rating: float

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]

@app.on_event("startup")
def startup_event():
    global model, df_data, all_movies, cold_start_recs, user_rated_movies
    
    # Define paths
    src_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(src_dir, "..", "data"))
    output_dir = os.path.abspath(os.path.join(src_dir, "..", "output"))
    
    model_path = os.path.join(data_dir, "svd_model.pkl")
    processed_csv_path = os.path.join(data_dir, "processed_movies.csv")
    cold_start_path = os.path.join(output_dir, "cold_start_recommendations.csv")
    
    print("Loading datasets and models...")
    
    # Load processed ratings and metadata
    if os.path.exists(processed_csv_path):
        df_data = pd.read_csv(processed_csv_path)
        all_movies = df_data[['movie_id', 'title']].drop_duplicates(subset=['movie_id']).to_dict(orient='records')
        
        # Build user history dictionary for faster lookup during recommendation
        user_rated_movies = df_data.groupby('user_id')['movie_id'].apply(set).to_dict()
    else:
        print(f"Warning: {processed_csv_path} not found.")
        df_data = pd.DataFrame()
        all_movies = []
        user_rated_movies = {}
        
    # Load SVD model
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        print(f"Warning: {model_path} not found. SVD recommendations will fail.")
        model = None
        
    # Load cold-start recommendations
    if os.path.exists(cold_start_path):
        df_cold = pd.read_csv(cold_start_path)
        cold_start_recs = df_cold.to_dict(orient='records')
    else:
        print(f"Warning: {cold_start_path} not found. Re-computing cold-start fallback...")
        if not df_data.empty:
            # Recompute on the fly
            movie_stats = df_data.groupby('movie_id').agg(
                average_rating=('rating', 'mean'),
                rating_count=('rating', 'count')
            ).reset_index()
            df_titles = df_data[['movie_id', 'title']].drop_duplicates(subset=['movie_id'])
            movie_stats = pd.merge(movie_stats, df_titles, on='movie_id')
            # Use same threshold (e.g. 50 ratings)
            popular = movie_stats[movie_stats['rating_count'] >= 50]
            if len(popular) < 10:
                popular = movie_stats
            sorted_popular = popular.sort_values(by=['average_rating', 'rating_count'], ascending=[False, False])
            cold_start_recs = sorted_popular.head(10).to_dict(orient='records')
        else:
            cold_start_recs = []
            
    print("Startup complete. API is ready.")

@app.get("/health")
def health_check():
    """Verify that the API is running and healthy."""
    return {"status": "ok"}

@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: int):
    """Retrieve top 10 movie recommendations for a user. Fallback to cold-start for new users."""
    global model, df_data, all_movies, cold_start_recs, user_rated_movies
    
    # 1. Cold-start check
    # If model/data is not loaded or user not in history, return cold-start recommendations
    if user_id not in user_rated_movies or model is None:
        recs = []
        for item in cold_start_recs[:10]:
            # Cold-start csv schema: movie_id, title, average_rating
            # API schema needs: movie_id, title, estimated_rating
            rating_val = item.get('average_rating', item.get('estimated_rating', 0.0))
            recs.append(RecommendationItem(
                movie_id=int(item['movie_id']),
                title=item['title'],
                estimated_rating=float(rating_val)
            ))
        return RecommendationResponse(user_id=user_id, recommendations=recs)
        
    # 2. Collaborative filtering with SVD
    rated_movie_ids = user_rated_movies[user_id]
    
    predictions = []
    for movie in all_movies:
        mid = int(movie['movie_id'])
        if mid not in rated_movie_ids:
            # Predict rating
            pred = model.predict(uid=user_id, iid=mid)
            predictions.append(RecommendationItem(
                movie_id=mid,
                title=movie['title'],
                estimated_rating=float(pred.est)
            ))
            
    # Sort predictions by estimated rating descending
    predictions.sort(key=lambda x: x.estimated_rating, reverse=True)
    
    return RecommendationResponse(user_id=user_id, recommendations=predictions[:10])
