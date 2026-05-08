from pathlib import Path
from typing import Tuple
import streamlit as st
import numpy as np
import pandas as pd

DATA_DIR = Path.cwd().parent / "data"
SUB_DIR = Path.cwd().parent / "submissions"

INTERACTIONS_PATH = DATA_DIR / "interactions_train.csv"
ITEMS_PATH = DATA_DIR / "items.csv"
ENRICHED_ITEMS_PATH = DATA_DIR / "enriched_items_merge_openlibrary_googlebooksAPI.csv" 
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

@st.cache_data
def load_catalog() -> pd.DataFrame:
    """[W.I.P (waiting to have the full full_items_dataset.csv file)]\\
    Loads the book catalog from a CSV file or database.

    :return: DataFrame containing the book catalog
    :rtype: pd.DataFrame
    """    
    # df = pd.read_csv(DATA_DIR / "full_item_dataset.csv")
    # return df[['i', 'Title', 'Author', 'Publisher', 'cover_url']]
    return pd.DataFrame({
        'i': [1, 2, 3, 4, 5],
        'Title': ['Dune', '1984', 'Foundation', 'Neuromancer', 'Hyperion'],
        'Author': ['Frank Herbert', 'George Orwell', 'Isaac Asimov', 'William Gibson', 'Dan Simmons'],
        'Publisher': ['Chilton Books', 'Secker & Warburg', 'Gnome Press', 'Ace', 'Doubleday'],
        # You will need actual URLs or paths for covers!
        'cover_url': ['https://via.placeholder.com/150x200?text=Dune'] * 5 
    })

def load_assets_basic():
    # Load Item Similarity
    item_sim = np.load(DATA_DIR / "item_similarity.npy", mmap_mode='r')
    
    # Load the historic user-item matrix
    historic_users = np.load(DATA_DIR / "full_data_mtx.npy", mmap_mode='r')
    
    # Load book catalog
    # books_df = pd.read_csv("data/book_catalog.csv", index_col='i')
    
    return item_sim, historic_users#, books_df

def load_assets_premium():
    similarity_matrix = np.load(DATA_DIR / "hybrid_item_similarity.npy", mmap_mode='r')
    return similarity_matrix