import streamlit as st
import pandas as pd
import time
# from data import DATA_DIR

def set_background(image_url):
    """
    Injects CSS to set the background image.
    
    :param image_url: URL or path to the background image
    :type image_url: str
    :return: None
    """
        
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

    # V2: If you want to add a semi-transparent overlay for better text readability, you can modify the CSS like this:
    # page_bg_img = f"""
    # <style>
    # .stApp {{
    #     background-image: url("{image_url}");
    #     background-size: cover;
    #     background-position: center;
    #     background-attachment: fixed;
    # }}
    # .stApp > header {{ background-color: transparent; }}
    # .block-container {{
    #     background-color: rgba(0, 0, 0, 0.7);
    #     padding: 2rem;
    #     border-radius: 10px;
    # }}
    # </style>
    # """
    # st.markdown(page_bg_img, unsafe_allow_html=True)


def door_animation():
    """Injects CSS to create a full-screen opening door effect."""
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
        animation-delay: 0.5s; /* Slight pause before opening */
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

def premium_model():
    """Dummy function to represent the heavy lifting."""
    time.sleep(2) # Simulating model computation time
    print("Premium model executed successfully in the background!")