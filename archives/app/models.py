import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.data import load_assets_basic, load_assets_premium
from typing import List


def basic_model(selected_books : List[str], catalog_df : pd.DataFrame) -> List[int]:
    item_sim, historic_users = load_assets_basic()
    num_items = item_sim.shape[0]
    # 1. Map selections to IDs
    read_book_ids = [catalog_df.Title.to_list().index(title) for title in selected_books]
    
    # 2. Create the interaction vector
    user_vector = np.zeros(num_items)
    user_vector[read_book_ids] = 1
    
    # A. ITEM-BASED PREDICTION
    item_scores = item_sim.dot(user_vector)
    
    # B. USER-BASED PREDICTION
    user_similarities = cosine_similarity(user_vector.reshape(1, -1), historic_users)
    user_scores = user_similarities.dot(historic_users).flatten()
    
    # C. HYBRID BLEND
    alpha = 0.24  # Weight for item-based vs user-based (found thourgh GridSearchCV) 
    hybrid_scores = (alpha * item_scores) + ((1 - alpha) * user_scores)

    top_10_ids = np.argsort(hybrid_scores)[-10:][::-1].tolist()

    return top_10_ids

def premium_model(selected_books : List[str], catalog_df : pd.DataFrame) -> List[int]:
    similarity_matrix = load_assets_premium()
    num_items = similarity_matrix.shape[0]
    read_book_ids = [catalog_df.Title.to_list().index(title) for title in selected_books]
    
    # Create a simple interaction vector for this new user (all 0s, with 1s for read books)
    user_vector = np.zeros(num_items)
    user_vector[read_book_ids] = 1
    
    # Multiply the similarity matrix by the user's vector
    # This is instantaneous! O(N) complexity instead of O(N^2)
    scores = similarity_matrix.dot(user_vector)
    
    # Set the scores of already read books to -1 so they don't get recommended
    scores[read_book_ids] = -1
    
    # Get the top 10 highest scoring unread books
    top_10_ids = np.argsort(scores)[-10:][::-1].tolist()
    return top_10_ids