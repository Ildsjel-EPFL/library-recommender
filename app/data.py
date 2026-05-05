from pathlib import Path
from typing import Tuple
import pandas as pd

DATA_DIR = Path.cwd().parent / "data"
SUB_DIR = Path.cwd().parent / "submissions"

INTERACTIONS_PATH = DATA_DIR / "interactions_train.csv"
ITEMS_PATH = DATA_DIR / "items.csv"
ENRICHED_ITEMS_PATH = DATA_DIR / "enriched_items_merge_openlibrary_googlebooksAPI.csv" 
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

def df_loader() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    """
    Load the interactions, items, and enriched items dataframes, and return them along with the number of unique users and items.   
    
    :return: A tuple containing the interactions dataframe, items dataframe, enriched items dataframe, number of unique users, and number of unique items.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]"""
    interactions = pd.read_csv(INTERACTIONS_PATH)
    interactions.columns = ['u', 'i', 't']
    items = pd.read_csv(ITEMS_PATH)
    df = pd.read_csv(ENRICHED_ITEMS_PATH)

    n_users = interactions['u'].nunique()
    n_items = items['i'].nunique()
    return interactions, items, df, n_users, n_items