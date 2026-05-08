import streamlit as st
import pandas as pd
import os
import gdown
import numpy as np

from utils.helpers import set_background
from utils.data import load_catalog, DATA_DIR
from utils.pop_ups import cookie_popup, premium_popup
from utils.models import basic_model, premium_model

st.set_page_config(page_title="The Ultimate Book Recommender", layout="wide")

# Initialize our session states
if "cookies_accepted" not in st.session_state:
    st.session_state.cookies_accepted = False
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "trigger_rickroll" not in st.session_state:
    st.session_state.trigger_rickroll = False

@st.cache_data(show_spinner="Downloading data from Google Drive... This only happens once!")
def load_data():
    ITEM_SIM_ID = "https://drive.google.com/file/d/1yPC8D1nLAcQ_Uzenx8iRXrKzDfLCpzJR/view?usp=sharing" 
    DATA_MTX_ID = "https://drive.google.com/file/d/1AZiKe2ArhSAKSDl3p5T17PnC5r8imgSz/view?usp=sharing"
    HYBRID_ITEM_SIM_ID = "https://drive.google.com/file/d/143JXstEzTcdokhwDNqEgIfjS7gvs0fF6/view?usp=sharing"

    # Define local file paths
    os.makedirs("data", exist_ok=True)
    item_sim_path = "data/item_similarity.npy"
    data_mtx_path = "data/full_data_mtx.npy"
    hybrid_item_similarity_path = "data/enriched_items_merge_openlibrary_googlebooksAPI.csv"

    # Download Item Similarity if it doesn't exist
    if not os.path.exists(item_sim_path):
        gdown.download(id=ITEM_SIM_ID, output=item_sim_path, quiet=False)

    # Download User-Item Matrix if it doesn't exist
    if not os.path.exists(data_mtx_path):
        gdown.download(id=DATA_MTX_ID, output=data_mtx_path, quiet=False)

    # Download Catalog if it doesn't exist
    if not os.path.exists(hybrid_item_similarity_path):
        gdown.download(id=HYBRID_ITEM_SIM_ID, output=hybrid_item_similarity_path, quiet=False)

    # Load the downloaded files
    # Use mmap_mode='r' for the massive .npy files to save RAM
    item_sim = np.load(item_sim_path, mmap_mode='r')
    historic_users = np.load(data_mtx_path, mmap_mode='r')
    hybrid_item_similarity = pd.read_csv(hybrid_item_similarity_path, mmap_mode='r')

    df_catalog = pd.read_csv(DATA_DIR / "enriched_items_merge_openlibrary_googlebooksAPI.csv", index_col='i')

    return item_sim, historic_users, hybrid_item_similarity, df_catalog

item_sim, historic_users, hybrid_item_similarity, df_catalog = load_data()

# Enforce Cookies first
if not st.session_state.cookies_accepted:
    cookie_popup()
    st.stop()

# STATE 1: Registration / Login
if not st.session_state.logged_in:
    # Set Background 1 (Replace with your actual URL or Base64 string)
    set_background("https://images.unsplash.com/photo-1507842217343-583bb7270b66?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80")
    
    st.title("Welcome to the Recommender")
    st.subheader("Please register to continue")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Register & Login")
        
        if submitted:
            if username and password:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Please enter both a username and password.")

# STATE 2: Selection & Prediction
elif st.session_state.predictions is None:
    # Still using Background 1
    set_background("https://images.unsplash.com/photo-1507842217343-583bb7270b66?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80")
    
    st.title("Find Your Next Great Read")
    
    book_options = df_catalog['Title'] + " by " + df_catalog['Author']
    selected_books = st.multiselect(
        "Select the last 3 books you read:", 
        options=book_options,
        max_selections=3
    )
    
    model_choice = st.radio("Choose your AI Model:", ["Basic (Free)", "Next-Gen (Premium)"])
    
    if st.button("Get Recommendations"):
        if len(selected_books) != 3:
            st.warning("Please select exactly 3 books.")
        elif model_choice == "Next-Gen (Premium)":
            premium_popup(df_catalog)
        else:
            with st.spinner("Calculating via Basic Model"):
                
                top_10_ids = basic_model()  # This function should return the top 10 recommended item IDs based on the selected books
                # 3. Store results in Session State to display in State 3
                st.session_state.predictions = df_catalog.loc[top_10_ids]
                st.rerun()

# STATE 3: Results Display
else:
    # Set Background 2 (Replace with your actual URL)
    set_background("https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80")
    
    st.title("Your Top 10 Recommendations")
    if st.button("Start Over"):
        st.session_state.predictions = None
        st.rerun()
        
    st.markdown("---")
    
    # Display the results in a grid (2 columns wide for better aesthetics)
    results_df = st.session_state.predictions
    
    for i in range(0, len(results_df), 2):
        cols = st.columns(2)
        
        # Book 1 in this row
        if i < len(results_df):
            book = results_df.iloc[i]
            with cols[0]:
                sub_cols = st.columns([1, 2])
                with sub_cols[0]:
                    st.image(book['Cover_URL'], use_column_width=True)
                with sub_cols[1]:
                    st.subheader(f"#{i+1}: {book['Title']}")
                    st.write(f"**Author:** {book['Author']}")
                    st.write(f"**Publisher:** {book['Publisher']}")
        
        # Book 2 in this row
        if i + 1 < len(results_df):
            book = results_df.iloc[i+1]
            with cols[1]:
                sub_cols = st.columns([1, 2])
                with sub_cols[0]:
                    st.image(book['Cover_URL'], use_column_width=True)
                with sub_cols[1]:
                    st.subheader(f"#{i+2}: {book['Title']}")
                    st.write(f"**Author:** {book['Author']}")
                    st.write(f"**Publisher:** {book['Publisher']}")
        
        st.markdown("---")