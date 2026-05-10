import streamlit as st
import pandas as pd
import time
from typing import List
from app.models import premium_model # Import the model to use inside the popup

def set_background(image_url):
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
    @keyframes openDoor { 100% { width: 0vw; opacity: 0; display: none; } }
    </style>
    <div class="door-left"></div>
    <div class="door-right"></div>
    """
    st.markdown(door_css, unsafe_allow_html=True)

@st.dialog("The Covenant of Lingering Echoes (Mandatory Cookie Policy)", icon="🍪")
def cookie_popup():
    st.write("To better illuminate thy path through these shifting archives, the Library must gather the crumbs of thy presence. These spectral memories allow the vellum to recognize thy spirit and the shadows to align with thy will. Wilt thou bind these fragments to thy journey, or shalt thou walk as a phantom, unremembered by the stone?")
    if st.button("I Accept (Like I have a choice)"):
        st.session_state.cookies_accepted = True
        st.rerun()

@st.dialog("💸 Premium Subscription Required")
def premium_popup(read_book_ids: List[int], df_catalog: pd.DataFrame, hybrid_item_similarity):
    st.write("To commune with the Exalted Archive, a living covenant must be struck; only through a recurring tribute of forty and two gilded discs shall the seal be broken and the high oracle’s voice remain unmuted for thy journey.")
    st.write("Strike the final mark to fulfill thy tithe and bind the golden covenant.")
    
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    html_button = f"""
        <a href="{youtube_url}" target="_blank" 
           style="display: inline-block; padding: 0.5em 1em; color: white; 
                  background-color: #FF4B4B; text-decoration: none; border-radius: 4px; font-weight: bold; text-align: center;">
            Continue to Payment
        </a>
    """
    st.markdown(html_button, unsafe_allow_html=True)
    
    with st.spinner("Processing Payment..."):
        time.sleep(5) 
        # Pass the matrix to the model
        top_10_ids = premium_model(read_book_ids, hybrid_item_similarity) 
        st.session_state.predictions = df_catalog.loc[top_10_ids]
        st.rerun() 
    
    if st.button("Cancel & Use Basic Model"):
        st.rerun()