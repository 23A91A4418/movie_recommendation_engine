import numpy as np
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, SVD, accuracy
from surprise.model_selection import train_test_split

def compute_ranking_metrics(predictions, k=10, threshold=3.5):
    """
    Compute average Precision@k and NDCG@k over all users in the test set.
    Relevance is defined as actual rating >= threshold.
    """
    # Map predictions to each user
    user_predictions = {}
    for uid, _, true_r, est, _ in predictions:
        if uid not in user_predictions:
            user_predictions[uid] = []
        user_predictions[uid].append((est, true_r))
        
    precisions = []
    ndcgs = []
    
    for uid, ratings in user_predictions.items():
        if not ratings:
            continue
            
        # Sort predictions by estimated rating descending
        ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = ratings[:k]
        
        # 1. Precision@k: fraction of top k items that are relevant
        rel_in_top_k = sum(1 for est, true_r in top_k if true_r >= threshold)
        precision = rel_in_top_k / k
        precisions.append(precision)
        
        # 2. NDCG@k
        # DCG@k
        dcg = sum(
            (1.0 if true_r >= threshold else 0.0) / np.log2(idx + 2)
            for idx, (est, true_r) in enumerate(top_k)
        )
        
        # IDCG@k: ideal sorting based on true ratings descending
        ideal_ratings = sorted(ratings, key=lambda x: x[1], reverse=True)[:k]
        idcg = sum(
            (1.0 if true_r >= threshold else 0.0) / np.log2(idx + 2)
            for idx, (est, true_r) in enumerate(ideal_ratings)
        )
        
        if idcg > 0:
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)
            
    return float(np.mean(precisions)), float(np.mean(ndcgs))

def evaluate_models(df):
    """
    Split dataset 80/20, evaluate User-Based CF and SVD,
    and return a dictionary with the evaluation results.
    """
    # Load dataset into Surprise format
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    
    # Train/test split (80% train, 20% test, random seed = 42)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # --- User-Based CF ---
    print("Evaluating User-Based Collaborative Filtering...")
    sim_options = {
        'name': 'cosine',
        'user_based': True
    }
    user_cf = KNNBasic(k=40, sim_options=sim_options, random_state=42, verbose=False)
    user_cf.fit(trainset)
    user_cf_preds = user_cf.test(testset)
    
    user_cf_rmse = float(accuracy.rmse(user_cf_preds, verbose=False))
    user_cf_prec, user_cf_ndcg = compute_ranking_metrics(user_cf_preds, k=10, threshold=3.5)
    
    # --- SVD ---
    print("Evaluating SVD Matrix Factorization...")
    svd = SVD(random_state=42)
    svd.fit(trainset)
    svd_preds = svd.test(testset)
    
    svd_rmse = float(accuracy.rmse(svd_preds, verbose=False))
    svd_prec, svd_ndcg = compute_ranking_metrics(svd_preds, k=10, threshold=3.5)
    
    results = {
        "user_based_cf": {
            "rmse": user_cf_rmse,
            "precision_at_10": user_cf_prec,
            "ndcg_at_10": user_cf_ndcg
        },
        "svd": {
            "rmse": svd_rmse,
            "precision_at_10": svd_prec,
            "ndcg_at_10": svd_ndcg
        }
    }
    
    return results
