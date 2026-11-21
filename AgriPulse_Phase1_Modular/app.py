"""AgriPulse Streamlit entry point.

Keep this file intentionally small: page configuration and application logic
live in the page and feature modules.
"""
import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
runpy.run_path(str(PROJECT_ROOT / "pages" / "home.py"), run_name="__main__")
