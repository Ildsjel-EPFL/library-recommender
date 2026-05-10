import streamlit as st
import pandas as pd
import numpy as np
import os
import gdown
import requests
import ast

PLACEHOLDER_COVER = "https://cdnattic.atticbooks.co.ke/img/Z665993.jpg"

@st.cache_resource(show_spinner="Downloading data from Google Drive... This only happens once!")
def load_data():
    ITEM_SIM_ID = "1yPC8D1nLAcQ_Uzenx8iRXrKzDfLCpzJR" 
    DATA_MTX_ID = "1AZiKe2ArhSAKSDl3p5T17PnC5r8imgSz"
    HYBRID_ITEM_SIM_ID = "143JXstEzTcdokhwDNqEgIfjS7gvs0fF6"
    items_csv_path = "data/enriched_items_merge_openlibrary_googlebooksAPI.csv"

    os.makedirs("data", exist_ok=True)
    item_sim_path = "data/item_similarity.npy"
    data_mtx_path = "data/full_data_mtx.npy"
    hybrid_item_similarity_path = "data/hybrid_item_similarity.npy"

    if not os.path.exists(item_sim_path):
        gdown.download(id=ITEM_SIM_ID, output=item_sim_path, quiet=False)
    if not os.path.exists(data_mtx_path):
        gdown.download(id=DATA_MTX_ID, output=data_mtx_path, quiet=False)
    if not os.path.exists(hybrid_item_similarity_path):
        gdown.download(id=HYBRID_ITEM_SIM_ID, output=hybrid_item_similarity_path, quiet=False)

    item_sim = np.load(item_sim_path, mmap_mode='r')
    historic_users = np.load(data_mtx_path, mmap_mode='r')
    hybrid_item_similarity = np.load(hybrid_item_similarity_path, mmap_mode='r')
    df_catalog = pd.read_csv(items_csv_path, index_col='i')

    return item_sim, historic_users, hybrid_item_similarity, df_catalog

@st.cache_data(show_spinner=False, ttl=86400)
def get_cover_on_the_fly(isbn_data):
    """Fetches a cover from OpenLibrary only when needed."""
    if pd.isna(isbn_data) or not isbn_data:
        return PLACEHOLDER_COVER
        
    if isinstance(isbn_data, str):
        if isbn_data.startswith('['):
            try:
                isbns = ast.literal_eval(isbn_data)
            except (ValueError, SyntaxError):
                isbns = []
        else:
            isbns = [i.strip() for i in isbn_data.split(';')]
    elif isinstance(isbn_data, list):
        isbns = isbn_data
    else:
        isbns = [str(isbn_data)]

    for isbn in isbns:
        clean_isbn = str(isbn).replace('-', '').replace(' ', '')
        if not clean_isbn:
            continue
            
        test_url = f"https://covers.openlibrary.org/b/isbn/{clean_isbn}-L.jpg?default=false"
        try:
            response = requests.head(test_url, timeout=2)
            if response.status_code == 200:
                return f"https://covers.openlibrary.org/b/isbn/{clean_isbn}-L.jpg"
        except requests.RequestException:
            continue
            
    return PLACEHOLDER_COVER