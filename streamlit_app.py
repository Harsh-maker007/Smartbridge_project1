"""
streamlit_app.py
=================
Root-level Streamlit entry point for Streamlit Community Cloud.
Streamlit Cloud looks for this file at the repo root.

Runs in DEMO MODE:
  - Emotion detection via Gemini API (no ML model files needed)
  - Rule-based keyword enhancement
  - Mixed emotion detection  
  - Gemini-powered personalized guidance
  - Full analytics dashboard
  - CSV interaction logging

To run locally:  streamlit run streamlit_app.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── NLTK data for cloud ───────────────────────────────────────
import nltk
for resource, name in [
    ("corpora/stopwords", "stopwords"),
    ("tokenizers/punkt",  "punkt"),
    ("corpora/wordnet",   "wordnet"),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(name, quiet=True)

# ── Streamlit page config ─────────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title="EmoLearn AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Gemini API key from Streamlit secrets (cloud) ────────
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# ── Custom CSS ────────────────────────────────────────────────
css_path = ROOT / "app" / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────
from streamlit_option_menu import option_menu

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 1rem 0;'>
            <div style='font-size: 2.5rem;'>🧠</div>
            <h2 style='color: #6C63FF; margin: 0; font-size: 1.3rem;'>EmoLearn AI</h2>
            <p style='color: #94A3B8; font-size: 0.75rem; margin-top: 0.25rem;'>
                Emotion-Aware Learning Platform
            </p>
            <span style='background:rgba(108,99,255,0.2); color:#8B83FF; 
                         padding:2px 10px; border-radius:12px; font-size:0.7rem;
                         border:1px solid rgba(108,99,255,0.4);'>
                ✨ Demo Mode
            </span>
        </div>
        <hr style='border-color: #2D3748; margin: 0.75rem 0;'>
        """,
        unsafe_allow_html=True,
    )

    selected = option_menu(
        menu_title=None,
        options=["🏠 Detect", "📊 Analytics", "⚖️ Compare", "ℹ️ About"],
        icons=["house-fill", "bar-chart-fill", "sliders", "info-circle-fill"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "#1A1A2E"},
            "icon":       {"color": "#6C63FF", "font-size": "16px"},
            "nav-link": {
                "font-size": "14px", "color": "#CBD5E0",
                "border-radius": "8px", "margin": "2px 0",
                "--hover-color": "#2D3748",
            },
            "nav-link-selected": {
                "background-color": "#6C63FF", "color": "white", "font-weight": "600",
            },
        },
    )

    st.markdown("<hr style='border-color:#2D3748;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748B;font-size:0.7rem;text-align:center;'>"
        "Gemini AI + Rule-Based Engine<br>© 2024 Smartbridge</p>",
        unsafe_allow_html=True,
    )

# ── Route Pages ───────────────────────────────────────────────
if "🏠 Detect" in selected:
    from app.pages.home_demo import render_home_demo
    render_home_demo()

elif "📊 Analytics" in selected:
    from app.pages.analytics import render_analytics
    render_analytics()

elif "⚖️ Compare" in selected:
    from app.pages.model_comparison import render_comparison
    render_comparison()

elif "ℹ️ About" in selected:
    from app.pages.about import render_about
    render_about()
