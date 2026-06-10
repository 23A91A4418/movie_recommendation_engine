import os
import zipfile
import urllib.request
import pandas as pd

# Constants
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = os.path.join(DATA_DIR, "ml-100k.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "ml-100k")
PROCESSED_CSV_PATH = os.path.join(DATA_DIR, "processed_movies.csv")

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

def download_and_extract():
    """Download the MovieLens 100k dataset if not present, and unzip it."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check if files already exist
    required_files = [
        os.path.join(EXTRACT_DIR, "ml-100k", "u.data"),
        os.path.join(EXTRACT_DIR, "ml-100k", "u.item")
    ]
    if all(os.path.exists(f) for f in required_files):
        print("Dataset already downloaded and extracted.")
        return
        
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading dataset from {ZIP_URL}...")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        print("Download complete.")
        
    print("Extracting files...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction complete.")

def preprocess_data():
    """Load u.data and u.item, merge, format genres, and save to processed_movies.csv."""
    # Note that extracted path has another folder 'ml-100k' inside
    dataset_path = os.path.join(DATA_DIR, "ml-100k")
    
    # Load ratings
    ratings_path = os.path.join(dataset_path, "u.data")
    print(f"Loading ratings from {ratings_path}...")
    df_ratings = pd.read_csv(
        ratings_path, 
        sep="\t", 
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": int, "movie_id": int, "rating": int, "timestamp": int}
    )
    
    # Load items
    items_path = os.path.join(dataset_path, "u.item")
    print(f"Loading items from {items_path}...")
    item_cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
    df_items = pd.read_csv(
        items_path,
        sep="|",
        names=item_cols,
        encoding="ISO-8859-1",
        dtype={"movie_id": int}
    )
    
    # Process genres column into pipe-separated string
    print("Formatting genres...")
    def get_genres_string(row):
        active_genres = [g for g in GENRES if row[g] == 1]
        return "|".join(active_genres)
        
    df_items["genres"] = df_items.apply(get_genres_string, axis=1)
    
    # Merge ratings and items
    print("Merging datasets...")
    df_merged = pd.merge(df_ratings, df_items[["movie_id", "title", "genres"]], on="movie_id")
    
    # Select columns
    df_final = df_merged[["user_id", "movie_id", "rating", "title", "genres"]]
    
    # Save to CSV
    print(f"Saving processed data to {PROCESSED_CSV_PATH}...")
    df_final.to_csv(PROCESSED_CSV_PATH, index=False)
    print("Preprocessing complete.")

if __name__ == "__main__":
    download_and_extract()
    preprocess_data()
