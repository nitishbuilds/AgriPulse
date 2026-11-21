"""AgriPulse Streamlit entry point. Run with: streamlit run app.py"""
import runpy
from pathlib import Path
import streamlit as st

# Keep page configuration before rendering any widgets.
st.set_page_config(page_title="AgriPulse MP", page_icon="🌾", layout="wide")

runpy.run_path(str(Path(__file__).parent / "pages" / "home.py"), run_name="__main__")
