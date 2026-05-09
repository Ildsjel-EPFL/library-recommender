import streamlit as st
import pandas as pd
import os
import gdown
import numpy as np
import requests
import ast
import time

from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# ==========================================
# SECTION 1: App Setup & Configuration
# ==========================================
st.set_page_config(page_title="The Ultimate Book Recommender", layout="wide")

# ==========================================
# SECTION 2: Session State Initialization
# ==========================================
# Session state variables act as the "memory" of our app. 
# We use them to remember where the user is in the application flow.
if "cookies_accepted" not in st.session_state:
    st.session_state.cookies_accepted = False
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "just_registered" not in st.session_state:  # Added to track the door animation
    st.session_state.just_registered = False

# ==========================================
# SECTION 3: Data Loading & Caching
# ==========================================
# We use @st.cache_resource so these massive files are only downloaded 
# and loaded into RAM once when the server starts.
@st.cache_resource(show_spinner="Downloading data from Google Drive... This only happens once!")
def load_data():
    ITEM_SIM_ID = "1yPC8D1nLAcQ_Uzenx8iRXrKzDfLCpzJR" 
    DATA_MTX_ID = "1AZiKe2ArhSAKSDl3p5T17PnC5r8imgSz"
    HYBRID_ITEM_SIM_ID = "143JXstEzTcdokhwDNqEgIfjS7gvs0fF6"
    items_csv_path = "data/enriched_items_merge_openlibrary_googlebooksAPI.csv"

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

    # Download Hybrid Matrix if it doesn't exist
    if not os.path.exists(hybrid_item_similarity_path):
        gdown.download(id=HYBRID_ITEM_SIM_ID, output=hybrid_item_similarity_path, quiet=False)

    # Load the downloaded files
    # Use mmap_mode='r' for the massive .npy files to prevent RAM crashes
    item_sim = np.load(item_sim_path, mmap_mode='r')
    historic_users = np.load(data_mtx_path, mmap_mode='r')
    hybrid_item_similarity = np.load(hybrid_item_similarity_path, mmap_mode='r')

    # Load the catalog
    df_catalog = pd.read_csv(items_csv_path, index_col='i')

    return item_sim, historic_users, hybrid_item_similarity, df_catalog

# Actually trigger the load_data function
item_sim, historic_users, hybrid_item_similarity, df_catalog = load_data()

# Add a placeholder image URL for books that truly have no cover anywhere
PLACEHOLDER_COVER = "https://cdnattic.atticbooks.co.ke/img/Z665993.jpg"

@st.cache_data(show_spinner=False, ttl=86400) # Cache clears after 24 hours
def get_cover_on_the_fly(isbn_data):
    """Fetches a cover from OpenLibrary only when needed."""
    if pd.isna(isbn_data) or not isbn_data:
        return PLACEHOLDER_COVER
        
    # Parse the ISBNs
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

    # Test the ISBNs against the API
    for isbn in isbns:
        clean_isbn = str(isbn).replace('-', '').replace(' ', '')
        if not clean_isbn:
            continue
            
        test_url = f"https://covers.openlibrary.org/b/isbn/{clean_isbn}-L.jpg?default=false"
        
        try:
            # Quick HEAD request to check if it exists
            response = requests.head(test_url, timeout=2)
            if response.status_code == 200:
                return f"https://covers.openlibrary.org/b/isbn/{clean_isbn}-L.jpg"
        except requests.RequestException:
            continue
            
    # If all ISBNs fail, return the placeholder
    return PLACEHOLDER_COVER

# ==========================================
# SECTION 4: UI & Aesthetic Functions
# ==========================================
def set_background(image_url):
    """Injects CSS to set the background image with a dark overlay."""
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

def door_animation():
    """Injects CSS to create a full-screen opening door effect upon login."""
    door_css = """
    <style>
    .door-left, .door-right {
        position: fixed;
        top: 0;
        width: 50vw;
        height: 100vh;
        background-color: #111;
        z-index: 9999;
        animation: openDoor 2s ease-in-out forwards;
        animation-delay: 0.5s;
    }
    .door-left { left: 0; transform-origin: left; border-right: 2px solid #fff; }
    .door-right { right: 0; transform-origin: right; border-left: 2px solid #fff; }
    
    @keyframes openDoor {
        100% { width: 0vw; opacity: 0; display: none; }
    }
    </style>
    <div class="door-left"></div>
    <div class="door-right"></div>
    """
    st.markdown(door_css, unsafe_allow_html=True)

# ==========================================
# SECTION 5: Machine Learning Models
# ==========================================
def basic_model(read_book_ids: List[int]) -> List[int]:
    """Calculates recommendations using Item-Sim matrix and on-the-fly User-Sim."""
    num_items = item_sim.shape[0]
    
    # 1. NEW: Filter out books that don't exist in our interaction matrices
    valid_book_ids = [book_id for book_id in read_book_ids if book_id < num_items]
    
    # Create the interaction vector for the current user
    user_vector = np.zeros(num_items)
    
    # 2. NEW: Only assign 1s if there are valid books
    if valid_book_ids: 
        user_vector[valid_book_ids] = 1
    
    # Calculate scores
    item_scores = item_sim.dot(user_vector)
    user_similarities = cosine_similarity(user_vector.reshape(1, -1), historic_users)
    user_scores = user_similarities.dot(historic_users).flatten()
    
    # Blend and filter
    alpha = 0.24  
    hybrid_scores = (alpha * item_scores) + ((1 - alpha) * user_scores)
    
    # 3. NEW: Only mask out the valid books (the out-of-bounds ones aren't in this array anyway)
    if valid_book_ids:
        hybrid_scores[valid_book_ids] = -np.inf 

    # Return top 10 IDs
    return np.argsort(hybrid_scores)[-10:][::-1].tolist()

def premium_model(read_book_ids: List[int]) -> List[int]:
    """Calculates recommendations instantaneously using the pre-computed hybrid matrix."""
    num_items = hybrid_item_similarity.shape[0]
    
    user_vector = np.zeros(num_items)
    user_vector[read_book_ids] = 1
    
    scores = hybrid_item_similarity.dot(user_vector)
    # scores[read_book_ids] = -np.inf
    
    return np.argsort(scores)[-10:][::-1].tolist()

# ==========================================
# SECTION 6: Popups & Dialogs
# ==========================================
@st.dialog("🍪 Mandatory Cookie Policy 🍪")
def cookie_popup():
    """Forces the user to accept cookies before using the app."""
    st.write("We use cookies to track your reading habits, judge your taste in literature, and sell your data to alien overlords. By clicking accept, you agree to these (totally reasonable) terms.")
    if st.button("I Accept (Like I have a choice)"):
        st.session_state.cookies_accepted = True
        st.rerun()

@st.dialog("💸 Premium Subscription Required")
def premium_popup(read_book_ids: List[int], df_catalog: pd.DataFrame):
    """The paywall popup with the hidden Rickroll."""
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
        time.sleep(5)  # Simulate payment processing delay
        top_10_ids = premium_model(read_book_ids) 
        st.session_state.predictions = df_catalog.loc[top_10_ids]
        st.rerun() 
    
    if st.button("Cancel & Use Basic Model"):
        st.rerun()

# ==========================================
# SECTION 7: Main Application Flow (State Machine)
# ==========================================

# --- STATE 0: Enforce Cookies ---
if not st.session_state.cookies_accepted:
    # set_background("https://i1-c.pinimg.com/1200x/b2/26/fb/b226fbeb41e4d09dbfd366122585594c.jpg")
    cookie_popup()
    st.stop()

# --- STATE 1: Registration / Login ---
if not st.session_state.logged_in:
    # set_background("https://images.unsplash.com/photo-1507842217343-583bb7270b66?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80")
    set_background("https://static.wikia.nocookie.net/clair-obscur/images/8/84/Coe33_coral_strange_door.jpg/revision/latest/scale-to-width-down/1000?cb=20250430012308")
    
    st.title("Welcome to the Recommender")
    st.subheader("Please register to continue")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Register & Login")
        
        if submitted:
            if username and password:
                st.session_state.logged_in = True
                st.session_state.just_registered = True # Set flag for animation
                st.rerun()
            else:
                st.error("Please enter both a username and password.")

# --- STATE 2: Selection & Prediction ---
elif st.session_state.predictions is None:
    set_background("https://images.unsplash.com/photo-1507842217343-583bb7270b66?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80")
    
    # Trigger the CSS animation if they just logged in
    if st.session_state.just_registered:
        door_animation()
        st.session_state.just_registered = False

    st.title("Find Your Next Great Read : "+f"{df_catalog.index[-1]}, {item_sim.shape[0]}")
    
    # Clean the catalog and create dropdown options
    clean_catalog = df_catalog.dropna(subset=['Title', 'Author'])
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
            read_book_ids = [book_options_dict[string] for string in selected_book_strings]
            
            if model_choice == "Next-Gen (Premium)":
                premium_popup(read_book_ids, df_catalog)
            else:
                with st.spinner("Calculating via Basic Model"):
                    top_10_ids = basic_model(read_book_ids) 
                    st.session_state.predictions = df_catalog.loc[top_10_ids]
                    st.rerun()

# --- STATE 3: Results Display ---
else:
    set_background("https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80")
    
    st.title("Your Top 10 Recommendations")
    
    if st.button("Start Over"):
        st.session_state.predictions = None
        st.rerun()
        
    st.markdown("---")
    
    results_df = st.session_state.predictions
    
    # Display the results in a 2-column grid
    for i in range(0, len(results_df), 2):
        cols = st.columns(2)
        
        # Book 1 in this row
        if i < len(results_df):
            book = results_df.iloc[i]
            with cols[0]:
                sub_cols = st.columns([1, 2])
                with sub_cols[0]:
                    # --- ON-THE-FLY CHECK ---
                    cover_to_display = book['cover_url']
                    if pd.isna(cover_to_display) or not str(cover_to_display).strip():
                        # Replace 'isbn' below with your actual ISBN column name!
                        cover_to_display = get_cover_on_the_fly(book['ISBN Valid'])
                    
                    st.image(cover_to_display, width='stretch')
                    # ------------------------
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
                    # --- ON-THE-FLY CHECK ---
                    cover_to_display = book['cover_url']
                    if pd.isna(cover_to_display) or not str(cover_to_display).strip():
                        # Replace 'isbn' below with your actual ISBN column name!
                        cover_to_display = get_cover_on_the_fly(book['ISBN Valid'])
                        
                    st.image(cover_to_display, width='stretch')
                    # ------------------------
                with sub_cols[1]:
                    st.subheader(f"#{i+2}: {book['Title']}")
                    st.write(f"**Author:** {book['Author']}")
                    st.write(f"**Publisher:** {book['Publisher']}")
        
        st.markdown("---")