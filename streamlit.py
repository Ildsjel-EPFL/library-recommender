import streamlit as st
import base64
from pathlib import Path
import pandas as pd

items_df = pd.read_csv("https://raw.githubusercontent.com/Ildsjel-EPFL/library-recommender/blob/main/data/items.csv")

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="centered")

def add_bg_from_url(image_url):
    """Adds a background image from a URL."""
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("{image_url}");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Set your custom background image URL here
# (If using a local image, it's highly recommended to convert it to base64 first)
BACKGROUND_URL = "https://images.unsplash.com/photo-1507842217343-583bb7270b66?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80"
add_bg_from_url(BACKGROUND_URL)

# ==========================================
# 2. MOCK DATA & MODEL FUNCTIONS
# ==========================================
# REPLACE THIS with your actual database of books
AVAILABLE_BOOKS = items_df["Title"].tolist()
AVAILABLE_BOOKS.sort()

def get_recommendations(book1, book2, book3):
    """
    REPLACE THIS FUNCTION WITH YOUR ML MODEL PREDICITONS.
    This mock function returns two fake recommendations.
    """
    return [
        {
            "title": "Fahrenheit 451 - Ray Bradbury",
            "cover_url": "https://covers.openlibrary.org/b/id/8259440-L.jpg"
        },
        {
            "title": "Brave New World - Aldous Huxley",
            "cover_url": "https://covers.openlibrary.org/b/id/8258567-L.jpg"
        }
    ]

# ==========================================
# 3. UI LAYOUT
# ==========================================

# Logo and Header
# Replace "https://via.placeholder.com/300x100?text=Your+Logo+Here" with your logo path (e.g., "logo.png")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("https://via.placeholder.com/300x100?text=Your+Logo+Here", use_container_width=True)

st.title("Discover Your Next Great Read 📖")
st.write("Tell us three books you've read and loved, and our AI will recommend what you should read next!")

st.markdown("---")

# User Input Section
st.subheader("Your Reading History")
book1 = st.selectbox("Book 1", AVAILABLE_BOOKS, key="b1")
book2 = st.selectbox("Book 2", AVAILABLE_BOOKS, key="b2")
book3 = st.selectbox("Book 3", AVAILABLE_BOOKS, key="b3")

# ==========================================
# 4. VALIDATION & RESULTS
# ==========================================
if st.button("Get Recommendations!", type="primary"):
    
    # Simple validation to ensure they selected actual books
    if "Select a book..." in [book1, book2, book3]:
        st.warning("⚠️ Please select three books to get a recommendation.")
    
    # Ensure they didn't select the same book multiple times
    elif len(set([book1, book2, book3])) < 3:
        st.warning("⚠️ Please select three distinct books.")
        
    else:
        st.success("Analyzing your taste...")
        
        # Call your ML model
        recommendations = get_recommendations(book1, book2, book3)
        
        st.markdown("### We think you'll love these:")
        
        # Display recommendations in a grid/columns
        cols = st.columns(len(recommendations))
        
        for index, col in enumerate(cols):
            with col:
                # Display cover and title
                st.image(recommendations[index]["cover_url"], use_container_width=True)
                st.write(f"**{recommendations[index]['title']}**")