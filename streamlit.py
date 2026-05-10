import streamlit as st
import pandas as pd

# Import from our custom modules
from app.data_loader import load_data, get_cover_on_the_fly
from app.models import basic_model
from app.ui import set_background, door_animation, cookie_popup, premium_popup

# --- App Setup ---
st.set_page_config(page_title="The Ultimate Book Recommender", layout="wide")

# --- Session State ---
if "cookies_accepted" not in st.session_state: st.session_state.cookies_accepted = False
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "predictions" not in st.session_state: st.session_state.predictions = None
if "just_registered" not in st.session_state: st.session_state.just_registered = False

# --- Load Data ---
item_sim, historic_users, hybrid_item_similarity, df_catalog = load_data()

# --- STATE 0: Enforce Cookies ---
if not st.session_state.cookies_accepted:
    set_background("https://i.pinimg.com/1200x/63/bb/ee/63bbee531be9b62c4396523d42e0c36e.jpg")
    cookie_popup()
    st.stop()

# --- STATE 1: Registration / Login ---
if not st.session_state.logged_in:
    set_background("https://github.com/Ildsjel-EPFL/library-recommender/blob/main/data/e33_doors.png?raw=true")
    st.title("Greetings, seeker of the hidden and the high.", text_alignment="center")
    st.subheader("Before you can step into the hallowed halls of literary wisdom, you must first prove your worth. Register your name and secret phrase to unlock the gates of knowledge and receive your personalized rune-song of book recommendations.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Register & Login"):
            if username and password:
                st.session_state.logged_in = True
                st.session_state.just_registered = True 
                st.rerun()
            else:
                st.error("Please enter both a username and password.")

# --- STATE 2: Selection & Prediction ---
elif st.session_state.predictions is None:
    set_background("https://static.wikia.nocookie.net/clair-obscur/images/e/e9/Coe33_Manor.png/revision/latest/scale-to-width-down/1200?cb=20250508134725")
    
    if st.session_state.just_registered:
        door_animation()
        st.session_state.just_registered = False

    st.title("Seek thou the destiny-bound chronicle that even now awakens to thy touch, for the vellum of thy next great legend awaits its unfolding.", text_alignment="center")
    
    clean_catalog = df_catalog.dropna(subset=["Title", "Author"])
    book_options = clean_catalog["Title"].astype(str) + " by " + clean_catalog["Author"].astype(str)
    book_options_dict = dict(zip(book_options, clean_catalog.index))
    
    selected_book_strings = st.multiselect("Select the last 3 books you read:", options=list(book_options_dict.keys()), max_selections=3)
    model_choice = st.radio("Choose your AI Model:", ["Basic (Free)", "Next-Gen (Premium)"])
    
    if st.button("Beseech the silent oracle of the hallowed stacks, that the shifting shadows may reveal the one true codex destined to illuminate the path of thy spirit."):
        if len(selected_book_strings) != 3:
            st.warning("Please select exactly 3 books.")
        else:
            read_book_ids = [book_options_dict[string] for string in selected_book_strings]
            
            if model_choice == "Next-Gen (Premium)":
                # Pass the required matrices into the popup
                premium_popup(read_book_ids, df_catalog, hybrid_item_similarity)
            else:
                with st.spinner("Calculating via Basic Model"):
                    # Pass the required matrices into the basic model
                    top_10_ids = basic_model(read_book_ids, item_sim, historic_users) 
                    st.session_state.predictions = df_catalog.loc[top_10_ids]
                    st.rerun()

# --- STATE 3: Results Display ---
else:
    set_background("https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80")
    st.title("Behold the Sacred Decad of the High Archives; ten echoes of truth plucked from the heart of the Great Silence, each a sovereign key to chambers long forgotten by the light of common day.", text_alignment="center")
    
    if st.button("Start Over"):
        st.session_state.predictions = None
        st.rerun()
        
    st.markdown("---")
    results_df = st.session_state.predictions
    
    for i in range(0, len(results_df), 2):
        cols = st.columns(2)
        
        if i < len(results_df):
            book = results_df.iloc[i]
            with cols[0]:
                sub_cols = st.columns([1, 2])
                with sub_cols[0]:
                    cover_to_display = book["cover_url"]
                    if pd.isna(cover_to_display) or not str(cover_to_display).strip():
                        cover_to_display = get_cover_on_the_fly(book["ISBN Valid"])
                    st.image(cover_to_display, use_container_width=True)
                with sub_cols[1]:
                    st.subheader(f"#{i+1}: {book['Title']}")
                    st.write(f"**Author:** {book['Author']}")
                    st.write(f"**Publisher:** {book['Publisher']}")
        
        if i + 1 < len(results_df):
            book = results_df.iloc[i+1]
            with cols[1]:
                sub_cols = st.columns([1, 2])
                with sub_cols[0]:
                    cover_to_display = book["cover_url"]
                    if pd.isna(cover_to_display) or not str(cover_to_display).strip():
                        cover_to_display = get_cover_on_the_fly(book["ISBN Valid"])
                    st.image(cover_to_display, use_container_width=True)
                with sub_cols[1]:
                    st.subheader(f"#{i+2}: {book['Title']}")
                    st.write(f"**Author:** {book['Author']}")
                    st.write(f"**Publisher:** {book['Publisher']}")
        st.markdown("---")