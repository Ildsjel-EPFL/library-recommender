import streamlit as st
import pandas as pd
import os
import gdown
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from typing import List

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

@st.cache_resource(show_spinner="Downloading data from Google Drive... This only happens once!")
def load_data():
    ITEM_SIM_ID = "1yPC8D1nLAcQ_Uzenx8iRXrKzDfLCpzJR" 
    DATA_MTX_ID = "1AZiKe2ArhSAKSDl3p5T17PnC5r8imgSz"
    HYBRID_ITEM_SIM_ID = "143JXstEzTcdokhwDNqEgIfjS7gvs0fF6"

    # Define local file paths
    os.makedirs("data", exist_ok=True)
    item_sim_path = "data/item_similarity.npy"
    data_mtx_path = "data/full_data_mtx.npy"
    hybrid_item_similarity_path = "data/hybrid_item_similarity.npy"

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
    hybrid_item_similarity = np.load(hybrid_item_similarity_path, mmap_mode='r')

    df_catalog = pd.read_csv("data/enriched_items_merge_openlibrary_googlebooksAPI.csv", index_col='i')

    return item_sim, historic_users, hybrid_item_similarity, df_catalog

item_sim, historic_users, hybrid_item_similarity, df_catalog = load_data()

def set_background(image_url):
    """
    Injects CSS to set the background image.
    
    :param image_url: URL or path to the background image
    :type image_url: str
    :return: None
    """
        
    # page_bg_img = f"""
    # <style>
    # .stApp {{
    #     background-image: url("{image_url}");
    #     background-size: cover;
    #     background-position: center;
    #     background-attachment: fixed;
    # }}
    # /* Adding a dark overlay so text remains readable */
    # .stApp > header {{
    #     background-color: transparent;
    # }}
    # .block-container {{
    #     background-color: rgba(0, 0, 0, 0.7);
    #     padding: 2rem;
    #     border-radius: 10px;
    # }}
    # </style>
    # """
    # st.markdown(page_bg_img, unsafe_allow_html=True)

    # V2: If you want to add a semi-transparent overlay for better text readability, you can modify the CSS like this:
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stApp > header {{ background-color: transparent; }}
    .block-container {{
        background-color: rgba(0, 0, 0, 0.7);
        padding: 2rem;
        border-radius: 10px;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def basic_model(read_book_ids: List[int]) -> List[int]:
    """
    Calculates recommendations using Item-Sim matrix and on-the-fly User-Sim.
    """
    num_items = item_sim.shape[0]
    
    # 1. Create the interaction vector
    user_vector = np.zeros(num_items)
    user_vector[read_book_ids] = 1
    
    # 2. ITEM-BASED PREDICTION
    item_scores = item_sim.dot(user_vector)
    
    # 3. USER-BASED PREDICTION
    user_similarities = cosine_similarity(user_vector.reshape(1, -1), historic_users)
    user_scores = user_similarities.dot(historic_users).flatten()
    
    # 4. HYBRID BLEND
    alpha = 0.24  # Weight found through GridSearchCV 
    hybrid_scores = (alpha * item_scores) + ((1 - alpha) * user_scores)

    # 5. Filter out the books the user just selected so they aren't recommended
    hybrid_scores[read_book_ids] = -np.inf

    # 6. Get the top 10 highest scoring unread books
    top_10_ids = np.argsort(hybrid_scores)[-10:][::-1].tolist()

    return top_10_ids

def premium_model(read_book_ids: List[int]) -> List[int]:
    """
    Calculates recommendations instantaneously using the pre-computed hybrid matrix.
    """
    num_items = hybrid_item_similarity.shape[0]
    
    # 1. Create the interaction vector
    user_vector = np.zeros(num_items)
    user_vector[read_book_ids] = 1
    
    # 2. Fast dot product against the hybrid matrix
    scores = hybrid_item_similarity.dot(user_vector)
    
    # 3. Filter out the books the user just selected
    scores[read_book_ids] = -np.inf
    
    # 4. Get the top 10 highest scoring unread books
    top_10_ids = np.argsort(scores)[-10:][::-1].tolist()
    
    return top_10_ids

@st.dialog("🍪 Mandatory Cookie Policy 🍪")
def cookie_popup():
    st.write("We use cookies to track your reading habits, judge your taste in literature, and sell your data to alien overlords. By clicking accept, you agree to these (totally reasonable) terms.")
    if st.button("I Accept (Like I have a choice)"):
        st.session_state.cookies_accepted = True
        st.rerun()

@st.dialog("💸 Premium Subscription Required")
def premium_popup(read_book_ids: List[int], df_catalog: pd.DataFrame):
    st.write("Our Premium Model requires an active subscription of $42/month.")
    st.write("Click continue to complete your payment.")
    
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    html_button = f"""
        <a href="{youtube_url}" target="_blank" 
           style="display: inline-block; padding: 0.5em 1em; color: white; 
                  background-color: #FF4B4B; text-decoration: none; 
                  border-radius: 4px; font-weight: bold; text-align: center;">
            Continue to Payment
        </a>
    """
    st.markdown(html_button, unsafe_allow_html=True)
    
    with st.spinner("Processing Payment..."):
        # Pass the extracted IDs directly into the model
        top_10_ids = premium_model(read_book_ids) 
        
        # Save to session state and rerun to trigger State 3 (Results Display)
        st.session_state.predictions = df_catalog.loc[top_10_ids]
        st.rerun() 
    
    if st.button("Cancel & Use Basic Model"):
        st.rerun()

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
    set_background("https://images.unsplash.com/photo-1507842217343-583bb7270b66?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80")
    
    st.title("Find Your Next Great Read")
    
    # 1. Clean the catalog to ensure no missing titles/authors crash the UI
    clean_catalog = df_catalog.dropna(subset=['Title', 'Author'])
    
    # 2. Vectorized dictionary creation (100x faster and safer than iterrows)
    book_options = clean_catalog['Title'].astype(str) + " by " + clean_catalog['Author'].astype(str)
    book_options_dict = dict(zip(book_options, clean_catalog.index))
    
    selected_book_strings = st.multiselect(
        "Select the last 3 books you read:", 
        options=list(book_options_dict.keys()),
        max_selections=3
    )
    
    model_choice = st.radio("Choose your AI Model:", ["Basic (Free)", "Next-Gen (Premium)"])
    
    if st.button("Get Recommendations"):
        if len(selected_book_strings) != 3:
            st.warning("Please select exactly 3 books.")
        else:
            # Convert the selected strings directly into a list of integer IDs
            read_book_ids = [book_options_dict[string] for string in selected_book_strings]
            
            if model_choice == "Next-Gen (Premium)":
                premium_popup(read_book_ids, df_catalog)
            else:
                with st.spinner("Calculating via Basic Model"):
                    top_10_ids = basic_model(read_book_ids) 
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
                    st.image(book['cover_url'], use_container_width=True)
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
                    st.image(book['cover_url'], use_container_width=True)
                with sub_cols[1]:
                    st.subheader(f"#{i+2}: {book['Title']}")
                    st.write(f"**Author:** {book['Author']}")
                    st.write(f"**Publisher:** {book['Publisher']}")
        
        st.markdown("---")