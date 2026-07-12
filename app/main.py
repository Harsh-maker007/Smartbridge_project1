"""
app/main.py
============
Streamlit application entry point.
Configures the page, loads sidebar navigation, and routes to pages.

Run with: streamlit run app/main.py
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from streamlit_option_menu import option_menu

# ── Page Config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="EmoLearn AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Sidebar Navigation ────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 1rem 0;'>
            <div style='font-size: 2.5rem;'>🧠</div>
            <h2 style='color: #6C63FF; margin: 0; font-size: 1.3rem;'>EmoLearn AI</h2>
            <p style='color: #94A3B8; font-size: 0.75rem; margin-top: 0.25rem;'>
                Emotion-Aware Learning Platform
            </p>
        </div>
        <hr style='border-color: #2D3748; margin: 0.5rem 0 1rem 0;'>
        """,
        unsafe_allow_html=True,
    )

    selected = option_menu(
        menu_title=None,
        options=["🏠 Detect", "📊 Analytics", "⚖️ Compare", "ℹ️ About"],
        icons=["house-fill", "bar-chart-fill", "sliders", "info-circle-fill"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "#1A1A2E"},
            "icon":       {"color": "#6C63FF", "font-size": "16px"},
            "nav-link": {
                "font-size":       "14px",
                "color":           "#CBD5E0",
                "border-radius":   "8px",
                "margin":          "2px 0",
                "--hover-color":   "#2D3748",
            },
            "nav-link-selected": {
                "background-color": "#6C63FF",
                "color":            "white",
                "font-weight":      "600",
            },
        },
    )

    st.markdown("<hr style='border-color: #2D3748; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748B; font-size:0.7rem; text-align:center;'>"
        "BiLSTM + BERT + Gemini AI<br>© 2024 Smartbridge</p>",
        unsafe_allow_html=True,
    )

# ── Page Routing ──────────────────────────────────────────────
if "🏠 Detect" in selected:
    from app.pages.home import render_home
    render_home()

elif "📊 Analytics" in selected:
    from app.pages.analytics import render_analytics
    render_analytics()

elif "⚖️ Compare" in selected:
    from app.pages.model_comparison import render_comparison
    render_comparison()

elif "ℹ️ About" in selected:
    from app.pages.about import render_about
    render_about()
