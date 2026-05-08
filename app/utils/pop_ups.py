import streamlit as st
import pandas as pd
from models import premium_model

@st.dialog("🍪 Mandatory Cookie Policy 🍪")
def cookie_popup():
    st.write("We use cookies to track your reading habits, judge your taste in literature, and sell your data to alien overlords. By clicking accept, you agree to these (totally reasonable) terms.")
    if st.button("I Accept (Like I have a choice)"):
        st.session_state.cookies_accepted = True
        st.rerun()

@st.dialog("💎 Premium Subscription Required")
def premium_popup(df_catalog : pd.DataFrame):
    st.write("Our Premium Model requires an active subscription of $42/month.")
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
    with st.spinner("Processing Payment..."):
        top_10_ids = premium_model() # Run the python function
        st.rerun() # Reload page to inject JS
        st.subheader("You should read:")
        for book_id in top_10_ids:
            title = df_catalog.loc[book_id, 'Title']
            author = df_catalog.loc[book_id, 'Author']
            st.write(f"📖 **{title}** by {author}")
    
    if st.button("Cancel & Use Basic Model"):
        st.rerun()

