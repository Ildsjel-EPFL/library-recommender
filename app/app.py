import streamlit as st
import pandas as pd
import time

# ==========================================
# 1. PAGE CONFIG & SESSION STATE SETUP
# ==========================================
st.set_page_config(page_title="The Ultimate Book Recommender", layout="wide")

# Initialize our session states
if "cookies_accepted" not in st.session_state:
    st.session_state.cookies_accepted = False
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "predictions" not in st.session_state:
    st.session_state.predictions = None

# ==========================================
# 2. HELPER FUNCTIONS (CSS & Mocks)
# ==========================================
def set_background(image_url):
    """Injects CSS to set the background image."""
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    /* Adding a dark overlay so text remains readable */
    .stApp > header {{
        background-color: transparent;
    }}
    .block-container {{
        background-color: rgba(0, 0, 0, 0.7);
        padding: 2rem;
        border-radius: 10px;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# MOCK DATA: Replace this with your actual parquet/csv loading
@st.cache_data
def load_catalog():
    return pd.DataFrame({
        'i': [1, 2, 3, 4, 5],
        'Title': ['Dune', '1984', 'Foundation', 'Neuromancer', 'Hyperion'],
        'Author': ['Frank Herbert', 'George Orwell', 'Isaac Asimov', 'William Gibson', 'Dan Simmons'],
        'Publisher': ['Chilton Books', 'Secker & Warburg', 'Gnome Press', 'Ace', 'Doubleday'],
        # You will need actual URLs or paths for covers!
        'Cover_URL': ['https://via.placeholder.com/150x200?text=Dune'] * 5 
    })

df_catalog = load_catalog()

# ==========================================
# 3. POP-UPS & DIALOGS
# ==========================================
@st.dialog("🍪 Mandatory Cookie Policy")
def cookie_popup():
    st.write("We use cookies to track your reading habits, judge your taste in literature, and sell your data to alien overlords. By clicking accept, you agree to these totally reasonable terms.")
    if st.button("I Accept (Like I have a choice)"):
        st.session_state.cookies_accepted = True
        st.rerun()

@st.dialog("💎 Premium Subscription Required")
def premium_popup():
    st.write("Our Premium Model requires an active subscription of $99/month.")
    st.write("Click continue to complete your payment.")
    
    # Streamlit buttons cannot easily open new tabs. 
    # We use HTML to create an anchor tag styled to look like a button.
    # Replace the Rickroll link with your desired YouTube video!
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
    
    if st.button("Cancel & Use Basic Model"):
        st.rerun()

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================

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
    
    model_choice = st.radio("Choose your AI Model:", ["Basic (Free)", "Premium (Next-Gen)"])
    
    if st.button("Get Recommendations"):
        if len(selected_books) != 3:
            st.warning("Please select exactly 3 books.")
        elif model_choice == "Premium (Next-Gen)":
            premium_popup()
        else:
            with st.spinner("Calculating via Basic Model..."):
                time.sleep(1.5) # Fake loading time for dramatic effect
                
                # --- YOUR REAL PREDICTION LOGIC GOES HERE ---
                # For now, we mock the results by just returning 10 random books
                mock_predictions = df_catalog.sample(n=min(10, len(df_catalog)), replace=True)
                st.session_state.predictions = mock_predictions
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