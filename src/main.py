import os
import json
import pickle
import pandas as pd

from data_loader import download_and_extract, preprocess_data
import models
import evaluate

def main():
    # Setup directories
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(src_dir, ".."))
    data_dir = os.path.join(root_dir, "data")
    output_dir = os.path.join(root_dir, "output")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Project Setup and Data Preprocessing
    print("=== Step 1 & 2: Data Acquisition and Preprocessing ===")
    download_and_extract()
    preprocess_data()
    
    # Load processed data
    processed_csv = os.path.join(data_dir, "processed_movies.csv")
    print(f"Loading merged dataset from {processed_csv}...")
    df = pd.read_csv(processed_csv)
    
    # Step 2: Model Implementation and Recommendation Generation for user_id = 1
    print("\n=== Step 3: Model Training and Recommendation Generation (user_id = 1) ===")
    target_user_id = 1
    
    # 2.1 User-Based CF
    print("Training User-Based Collaborative Filtering...")
    user_cf_model = models.train_user_based_cf(df)
    user_cf_recs = models.get_cf_recommendations(user_cf_model, df, target_user_id, top_n=10)
    
    user_cf_df = pd.DataFrame(user_cf_recs)
    user_cf_csv = os.path.join(output_dir, "user_based_recommendations.csv")
    user_cf_df.to_csv(user_cf_csv, index=False)
    print(f"Saved User-Based CF recommendations to {user_cf_csv}")
    
    # 2.2 SVD (Matrix Factorization)
    print("Training SVD Matrix Factorization...")
    svd_model = models.train_svd(df)
    svd_recs = models.get_cf_recommendations(svd_model, df, target_user_id, top_n=10)
    
    svd_df = pd.DataFrame(svd_recs)
    svd_csv = os.path.join(output_dir, "svd_recommendations.csv")
    svd_df.to_csv(svd_csv, index=False)
    print(f"Saved SVD recommendations to {svd_csv}")
    
    # Pickle the trained SVD model for API deployment
    model_pickle_path = os.path.join(data_dir, "svd_model.pkl")
    with open(model_pickle_path, 'wb') as f:
        pickle.dump(svd_model, f)
    print(f"Saved trained SVD model to {model_pickle_path}")
    
    # 2.3 Content-Based Filtering
    print("Generating Content-Based recommendations...")
    content_recs = models.get_content_based_recommendations(df, target_user_id, top_n=10)
    
    content_df = pd.DataFrame(content_recs)
    content_csv = os.path.join(output_dir, "content_based_recommendations.csv")
    content_df.to_csv(content_csv, index=False)
    print(f"Saved Content-Based recommendations to {content_csv}")
    
    # 2.4 Cold-Start Recommendations
    print("Generating Cold-Start recommendations...")
    cold_recs = models.get_cold_start_recommendations(df, min_ratings=50, top_n=10)
    
    cold_df = pd.DataFrame(cold_recs)
    cold_csv = os.path.join(output_dir, "cold_start_recommendations.csv")
    cold_df.to_csv(cold_csv, index=False)
    print(f"Saved Cold-Start recommendations to {cold_csv}")
    
    # Step 3: Model Evaluation
    print("\n=== Step 4: Model Evaluation Pipeline ===")
    eval_metrics = evaluate.evaluate_models(df)
    
    eval_json_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(eval_json_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Saved evaluation metrics to {eval_json_path}")
    
    print("\n=== All Preprocessing, Training, and Evaluation complete! ===")

if __name__ == "__main__":
    main()
