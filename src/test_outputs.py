import os
import json
import pandas as pd

def test_outputs():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(src_dir, ".."))
    data_dir = os.path.join(root_dir, "data")
    output_dir = os.path.join(root_dir, "output")
    
    # Check processed data
    processed_path = os.path.join(data_dir, "processed_movies.csv")
    assert os.path.exists(processed_path), f"Missing: {processed_path}"
    df_proc = pd.read_csv(processed_path)
    expected_proc_cols = ["user_id", "movie_id", "rating", "title", "genres"]
    assert list(df_proc.columns) == expected_proc_cols, f"Invalid columns in processed_movies.csv: {df_proc.columns}"
    assert df_proc["user_id"].dtype in [int, "int64"], "user_id is not integer"
    assert df_proc["movie_id"].dtype in [int, "int64"], "movie_id is not integer"
    assert df_proc["rating"].dtype in [int, "int64"], "rating is not integer"
    assert df_proc["title"].dtype in [object, str], "title is not string"
    assert df_proc["genres"].dtype in [object, str], "genres is not string"
    print("[OK] data/processed_movies.csv looks perfect!")
    
    # Check user-based cf recommendations
    ub_path = os.path.join(output_dir, "user_based_recommendations.csv")
    assert os.path.exists(ub_path), f"Missing: {ub_path}"
    df_ub = pd.read_csv(ub_path)
    assert len(df_ub) == 10, f"Expected exactly 10 user recommendations, got {len(df_ub)}"
    expected_ub_cols = ["movie_id", "title", "estimated_rating"]
    assert list(df_ub.columns) == expected_ub_cols, f"Invalid columns in user_based_recommendations.csv: {df_ub.columns}"
    assert df_ub["movie_id"].dtype in [int, "int64"], "movie_id is not integer"
    assert df_ub["estimated_rating"].dtype in [float, "float64"], "estimated_rating is not float"
    print("[OK] output/user_based_recommendations.csv looks perfect!")
    
    # Check SVD recommendations
    svd_path = os.path.join(output_dir, "svd_recommendations.csv")
    assert os.path.exists(svd_path), f"Missing: {svd_path}"
    df_svd = pd.read_csv(svd_path)
    assert len(df_svd) == 10, f"Expected exactly 10 SVD recommendations, got {len(df_svd)}"
    expected_svd_cols = ["movie_id", "title", "estimated_rating"]
    assert list(df_svd.columns) == expected_svd_cols, f"Invalid columns in svd_recommendations.csv: {df_svd.columns}"
    assert df_svd["movie_id"].dtype in [int, "int64"], "movie_id is not integer"
    assert df_svd["estimated_rating"].dtype in [float, "float64"], "estimated_rating is not float"
    print("[OK] output/svd_recommendations.csv looks perfect!")
    
    # Check content-based recommendations
    cb_path = os.path.join(output_dir, "content_based_recommendations.csv")
    assert os.path.exists(cb_path), f"Missing: {cb_path}"
    df_cb = pd.read_csv(cb_path)
    assert len(df_cb) == 10, f"Expected exactly 10 content recommendations, got {len(df_cb)}"
    expected_cb_cols = ["movie_id", "title", "similarity_score"]
    assert list(df_cb.columns) == expected_cb_cols, f"Invalid columns in content_based_recommendations.csv: {df_cb.columns}"
    assert df_cb["movie_id"].dtype in [int, "int64"], "movie_id is not integer"
    assert df_cb["similarity_score"].dtype in [float, "float64"], "similarity_score is not float"
    print("[OK] output/content_based_recommendations.csv looks perfect!")
    
    # Check cold-start recommendations
    cs_path = os.path.join(output_dir, "cold_start_recommendations.csv")
    assert os.path.exists(cs_path), f"Missing: {cs_path}"
    df_cs = pd.read_csv(cs_path)
    assert len(df_cs) == 10, f"Expected exactly 10 cold-start recommendations, got {len(df_cs)}"
    expected_cs_cols = ["movie_id", "title", "average_rating"]
    assert list(df_cs.columns) == expected_cs_cols, f"Invalid columns in cold_start_recommendations.csv: {df_cs.columns}"
    assert df_cs["movie_id"].dtype in [int, "int64"], "movie_id is not integer"
    assert df_cs["average_rating"].dtype in [float, "float64"], "average_rating is not float"
    
    # Check sorting
    is_sorted = df_cs["average_rating"].is_monotonic_decreasing
    assert is_sorted, "cold-start recommendations are not sorted in descending order of average_rating"
    print("[OK] output/cold_start_recommendations.csv looks perfect!")
    
    # Check evaluation metrics JSON
    eval_path = os.path.join(output_dir, "evaluation_metrics.json")
    assert os.path.exists(eval_path), f"Missing: {eval_path}"
    with open(eval_path, 'r') as f:
        metrics = json.load(f)
        
    for key in ["user_based_cf", "svd"]:
        assert key in metrics, f"Missing '{key}' in evaluation metrics"
        model_metrics = metrics[key]
        for metric in ["rmse", "precision_at_10", "ndcg_at_10"]:
            assert metric in model_metrics, f"Missing metric '{metric}' for '{key}'"
            assert isinstance(model_metrics[metric], float), f"Metric '{metric}' for '{key}' is not a float"
            
    print("[OK] output/evaluation_metrics.json looks perfect!")
    print("\nAll output validation checks PASSED successfully!")

if __name__ == "__main__":
    test_outputs()
