import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple

from data import df_loader
from features_extraction import compute_embeddings
from baseline_functions import create_data_mtx, item_based_predict, create_submission

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# interactions, items, df, n_users, n_items = df_loader()
# embeddings = compute_embeddings(df, model_choice="e5", device=device)

def temporal_split(interactions : pd.DataFrame, train_pct = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the interactions DataFrame into training and validation sets based on a temporal split.
    
    :param interactions: The user-item interaction DataFrame with a timestamp column 't'.
    :type interactions: pd.DataFrame
    :param train_pct: The percentage of interactions to include in the training set (default is 0.8).
    :type train_pct: float, optional
    :return: A tuple containing the training and validation DataFrames.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    
    interactions.sort_values('t', inplace=True)
    split_idx = int(len(interactions) * train_pct)
    train_data = interactions.iloc[:split_idx]
    val_data = interactions.iloc[split_idx:]
    return train_data, val_data

def train(interaction_df : pd.DataFrame, text_item_similarity : npt.NDArray[np.float64], n_users : int, alpha: float, predict : bool = False) -> npt.NDArray[np.float64] :
    """
    Compute a hybrid item similarity matrix by combining collaborative filtering and content-based similarities.
    
    :param interactions: The user-item interaction DataFrame.
    :type interactions: pd.DataFrame
    :param embeddings: The precomputed item embeddings for content-based similarity.
    :type embeddings: npt.NDArray[np.float32]
    :param n_users: The number of unique users in the dataset.
    :type n_users: int
    :param alpha: The weight for combining collaborative filtering and content-based similarities (0 <= alpha <= 1).
    :type alpha: float
    :param predict: Whether to return predicted interaction scores instead of the similarity matrix (default is False).
    :type predict: bool, optional
    :return: A hybrid item similarity matrix.
    :rtype: npt.NDArray[np.float64]
    """

    data_mtx = create_data_mtx(interaction_df, n_users)
    cf_item_similarity = cosine_similarity(data_mtx.T)
    hybrid_item_similarity = (alpha * cf_item_similarity) + ((1 - alpha) * text_item_similarity)
    if predict:
        return item_based_predict(data_mtx, hybrid_item_similarity)
    return hybrid_item_similarity

def grid_search(alpha_min : float, alpha_max : float, alpha_step : float) -> Tuple[float, pd.DataFrame, npt.NDArray[np.float64], int]:
    """
    Perform a grid search to find the best alpha value for blending collaborative filtering and content-based similarities.
    
    :param alpha_min: The minimum alpha value to test.
    :type alpha_min: float
    :param alpha_max: The maximum alpha value to test.
    :type alpha_max: float
    :param alpha_step: The step size for alpha values to test.
    :type alpha_step: float
    :return: The best alpha value that yields the highest validation precision.
    :rtype: float
    """

    # Data Loading
    interactions, items, df, n_users, n_items = df_loader()
    embeddings = compute_embeddings(df, model_choice="e5", device=device)
    # Temporal Train/Val Split
    train_data, val_data = temporal_split(interactions)
    # Build ground-truth dictionaries for fast lookup
    val_histories = val_data.groupby('u')['i'].apply(set).to_dict()
    # We only evaluate users who actually have interactions in the validation set
    eval_users = list(val_histories.keys())
    # Compute Matrices on train data
    train_data_mtx = create_data_mtx(train_data, n_users, n_items)
    train_cf_sim = cosine_similarity(train_data_mtx.T)
    # Ensure embeddings tensor/array is ready
    text_sim = cosine_similarity(embeddings)
    # hyperparameter definition
    alphas_to_test = np.linspace(start=alpha_min, stop=alpha_max, num=alpha_step).tolist()
    best_alpha = 0.0
    best_precision = 0.0
    results = {}
    # Grid Search Loop
    for alpha in alphas_to_test:
        # Blend the matrices
        hybrid_sim = hybrid_sim = (alpha * train_cf_sim) + ((1 - alpha) * text_sim)
        # Generate predictions
        predictions = item_based_predict(train_data_mtx, hybrid_sim)
        # Evaluate Precision@10
        precisions = []
        for u in eval_users:
            u_scores = predictions[u, :].copy()
            # Get Top 10 recommendations
            top_10 = np.argsort(u_scores)[-10:][::-1]
            # Calculate how many of the top 10 are in the user's Validation set
            true_items = val_histories[u]
            hits = len(set(top_10).intersection(true_items))
            precisions.append(hits / 10.0)
        avg_precision = np.mean(precisions)
        results[alpha] = avg_precision
        print(f"Alpha: {alpha} | Validation Precision@10: {avg_precision:.5f}")
        if avg_precision > best_precision:
            best_precision = avg_precision
            best_alpha = alpha
    return best_alpha, interactions, text_sim, n_users

def main(sample_submission_df : pd.DataFrame, output_path : str, alpha_min : float, alpha_max : float = None, alpha_step : int = 1) -> None:
    """
    Main function to execute the recommendation pipeline: performs grid search for best alpha and generates submission file.
    :param sample_submission_df: A DataFrame containing the sample submission format with columns "u" for user IDs and "i" for item IDs.
    :type sample_submission_df: pd.DataFrame
    :param output_path: The file path to save the generated submission CSV.
    :type output_path: str
    :param alpha_min: The minimum alpha value to test in the grid search.
    :type alpha_min: float
    :param alpha_max: The maximum alpha value to test in the grid search.
    :type alpha_max: float 
    :param alpha_step: The step size for alpha values to test in the grid search.
    :type alpha_step: int 
    """
    alpha_max = alpha_min if alpha_max is None else alpha_max
    best_alpha, interactions, text_sim, n_users = grid_search(alpha_min, alpha_max, alpha_step)
    print(f"Best alpha found: {best_alpha}")
    create_submission(sample_submission_df, train(interaction_df=interactions, text_item_similarity=text_sim, n_users=n_users, alpha=best_alpha, predict=True), output_path)