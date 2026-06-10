# Movie Recommendation Engine

A comprehensive movie recommendation system built on the MovieLens 100k dataset. It implements multiple filtering techniques, evaluates their accuracy and ranking quality using industry-standard metrics, and serves real-time predictions via a containerized FastAPI web service.

## Project Structure

```
/ (repository root)
├── data/                      # Raw and processed data (datasets, models)
├── output/                    # Generated recommendation CSVs and evaluation metrics JSON
├── src/                       # Python source code
│   ├── data_loader.py         # Downloads, unzips, and processes the dataset
│   ├── models.py              # Model definitions (CF, SVD, Content-Based, Cold-Start)
│   ├── evaluate.py            # Train-test split (80/20) and evaluation pipeline
│   ├── api.py                 # FastAPI web application
│   ├── main.py                # Pipeline orchestrator
│   └── test_outputs.py        # Output validation script
├── Dockerfile                 # Docker container builder configuration
├── docker-compose.yml         # Container orchestration configuration
├── .env.example               # Environmental variables blueprint
├── .env                       # Local environment configurations
├── requirements.txt           # Python packages list
└── README.md                  # Documentation (this file)
```

## Implemented Recommenders

1. **User-Based Collaborative Filtering (KNN)**: Predicts a user's ratings based on the ratings history of similar users. Built using cosine similarity over $K$ nearest neighbors.
2. **Item-Based Collaborative Filtering (KNN)**: Measures similarity between movies and predicts user ratings based on similar movies they liked.
3. **Matrix Factorization (SVD)**: Decomposes the user-item interaction matrix into lower-dimensional latent user and movie factor matrices. Serves as the primary engine for the real-time API.
4. **Content-Based Filtering**: Represents movies as TF-IDF feature vectors of genres. Computes a user preference profile based on history and finds the most similar unrated movies using cosine similarity.
5. **Cold-Start Recommendations**: Recommends the most popular movies (highest average rating with at least 50 ratings) for new users with no historical ratings.

## Getting Started

### Prerequisites
- Docker & Docker Compose (for containerized deployment)
- Python 3.10 (for local execution)

### Configuration
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Default parameters:
- `API_PORT=8000`: Port where the FastAPI server is served.
- `DEFAULT_USER_ID=1`: Default user for recommendations.

### Option 1: Running with Docker Compose (Recommended)
You can run the entire pipeline (data download, training, and API startup) in one command:
```bash
docker-compose up --build
```
This automatically:
- Builds the application image.
- Downloads the MovieLens 100k dataset.
- Runs data preprocessing and trains the models.
- Generates all offline CSV recommendations and evaluation metrics.
- Starts the FastAPI server on `http://localhost:8000`.

### Option 2: Running Locally
1. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline orchestrator:
   ```bash
   python src/main.py
   ```
   This will download data, train the models, save recommendations, evaluate performance, and pickle the SVD model.
4. Start the API server:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

## Evaluation Metrics

The User-Based Collaborative Filtering and SVD models are evaluated using an 80/20 train/test split. The system computes three metrics:
- **RMSE (Root Mean Squared Error)**: Measures rating prediction accuracy.
- **Precision@10**: Evaluates the fraction of recommended items in the top-10 that are relevant (rating $\ge$ 3.5).
- **NDCG@10 (Normalized Discounted Cumulative Gain)**: Measures ranking quality, rewarding the system for placing highly relevant items at the top of the list.

Metrics are saved to `output/evaluation_metrics.json`.

## API Documentation

### 1. Health Check
Checks if the server is healthy and running.
- **Endpoint**: `GET /health`
- **Response**:
  ```json
  {
    "status": "ok"
  }
  ```

### 2. Retrieve Movie Recommendations
Retrieves the top-10 recommended movies for a given user using the SVD model. Automatically falls back to cold-start recommendations for unknown users.
- **Endpoint**: `GET /recommendations/{user_id}`
- **Response**:
  ```json
  {
    "user_id": 1,
    "recommendations": [
      {
        "movie_id": 12,
        "title": "Usual Suspects, The (1995)",
        "estimated_rating": 4.812
      },
      ...
    ]
  }
  ```