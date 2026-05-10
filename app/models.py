import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

def basic_model(read_book_ids: List[int], item_sim: np.ndarray, historic_users: np.ndarray) -> List[int]:
    """Calculates recommendations using Item-Sim matrix and on-the-fly User-Sim."""
    num_items = item_sim.shape[0]
    valid_book_ids = [book_id for book_id in read_book_ids if book_id < num_items]
    
    user_vector = np.zeros(num_items)
    if valid_book_ids: 
        user_vector[valid_book_ids] = 1
    
    item_scores = item_sim.dot(user_vector)
    user_similarities = cosine_similarity(user_vector.reshape(1, -1), historic_users)
    user_scores = user_similarities.dot(historic_users).flatten()
    
    alpha = 0.24  
    hybrid_scores = (alpha * item_scores) + ((1 - alpha) * user_scores)
    
    if valid_book_ids:
        hybrid_scores[valid_book_ids] = -np.inf 

    return np.argsort(hybrid_scores)[-10:][::-1].tolist()

def premium_model(read_book_ids: List[int], hybrid_item_similarity: np.ndarray) -> List[int]:
    """Calculates recommendations instantaneously using the pre-computed hybrid matrix."""
    num_items = hybrid_item_similarity.shape[0]
    
    user_vector = np.zeros(num_items)
    user_vector[read_book_ids] = 1
    
    scores = hybrid_item_similarity.dot(user_vector)
    # scores[read_book_ids] = -np.inf
    
    return np.argsort(scores)[-10:][::-1].tolist()