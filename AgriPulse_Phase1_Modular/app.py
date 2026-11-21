
import streamlit as st

# AgriPulse main page
home_page = st.Page(
    "pages/home.py",
    title="AgriPulse",
    default=True
)

# Hide Streamlit's automatic page navigation
navigation = st.navigation(
    [home_page],
    position="hidden"
)

# Run the main application
navigation.run()
