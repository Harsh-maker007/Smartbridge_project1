"""
app/pages/about.py
====================
About page — project overview, architecture, team, and tech stack.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st


def render_about():
    """Render the About page."""

    st.markdown(
        """
        <div style='text-align:center; padding: 1.5rem 0;'>
            <div style='font-size:4rem;'>🧠</div>
            <h1 style='font-size:2.2rem; font-weight:700; color:#E2E8F0; margin:0.5rem 0;'>
                EmoLearn AI
            </h1>
            <p style='color:#94A3B8; font-size:1rem;'>
                AI-Driven Emotion Detection & Personalized Learning Support Platform
            </p>
            <div style='margin-top:1rem;'>
                <span class="confidence-badge">v1.0.0</span>&nbsp;
                <span class="confidence-badge">Smartbridge AI/ML</span>&nbsp;
                <span class="confidence-badge">Python 3.10+</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Mission ───────────────────────────────────────────────
    st.markdown(
        """
        <div class="guidance-panel" style="text-align:center; margin-bottom:1.5rem;">
            <h3 style="color:#6C63FF;">🎯 Our Mission</h3>
            <p style="color:#CBD5E0; font-size:1rem; max-width:600px; margin:0 auto;">
                Every student learns differently and feels differently while learning.
                EmoLearn AI detects your emotional state from your own words and provides
                personalized guidance — because the right support at the right emotional moment
                can transform a frustrating hour into a breakthrough.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tech Stack ────────────────────────────────────────────
    st.markdown('<p class="section-header">🛠️ Technology Stack</p>', unsafe_allow_html=True)

    tech = [
        ("🔵", "BiLSTM",       "TensorFlow 2.15 / Keras",   "RNN-based emotion classifier"),
        ("🟢", "BERT",         "HuggingFace Transformers",   "Fine-tuned bert-base-uncased"),
        ("🤖", "Gemini AI",    "Google Generative AI",       "Personalized learning guidance"),
        ("🌐", "Streamlit",    "Streamlit 1.32",             "Interactive web application"),
        ("📊", "Visualization","Plotly + Seaborn",           "Interactive charts & analytics"),
        ("🐍", "Data Science", "Pandas + NumPy + sklearn",  "Data processing & ML utilities"),
    ]

    cols = st.columns(3)
    for i, (icon, name, tech_name, desc) in enumerate(tech):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="emotion-card" style="text-align:center; padding:1.2rem;">
                    <div style="font-size:2rem;">{icon}</div>
                    <div style="font-weight:700; color:#E2E8F0; margin:0.5rem 0;">{name}</div>
                    <div style="font-size:0.8rem; color:#6C63FF;">{tech_name}</div>
                    <div style="font-size:0.75rem; color:#94A3B8; margin-top:0.5rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Data Flow ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🔄 How It Works</p>', unsafe_allow_html=True)

    steps = [
        ("1️⃣", "Student Input",      "You describe your study problem in free text"),
        ("2️⃣", "BiLSTM Analysis",    "RNN model analyzes word sequences and patterns"),
        ("3️⃣", "BERT Analysis",      "Transformer model understands deep context"),
        ("4️⃣", "Keyword Matching",   "Rule-based system scans for emotion keywords"),
        ("5️⃣", "Ensemble Decision",  "Weighted combination (40% BiLSTM + 60% BERT)"),
        ("6️⃣", "Mixed Detection",    "Check if multiple emotions are present"),
        ("7️⃣", "Gemini Guidance",    "AI generates personalized learning advice"),
        ("8️⃣", "Log & Learn",        "Interaction saved for analytics and insights"),
    ]

    for i in range(0, len(steps), 4):
        cols = st.columns(4)
        for j, (icon, title, desc) in enumerate(steps[i:i+4]):
            with cols[j]:
                st.markdown(
                    f"""
                    <div class="emotion-card" style="padding:1rem; text-align:center;">
                        <div style="font-size:1.8rem;">{icon}</div>
                        <div style="font-weight:600; color:#E2E8F0; font-size:0.9rem; margin:0.5rem 0;">{title}</div>
                        <div style="font-size:0.75rem; color:#94A3B8;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Emotion Classes ───────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🎭 Detected Emotions</p>', unsafe_allow_html=True)

    from config.emotions import EMOTION_META, GUIDANCE_TONE
    emotions = ["bored", "confident", "confused", "curious", "frustrated"]
    cols = st.columns(5)
    for i, emotion in enumerate(emotions):
        meta = EMOTION_META[emotion]
        with cols[i]:
            st.markdown(
                f"""
                <div class="emotion-card" style="border-color:{meta['color_hex']}40; text-align:center; padding:1.2rem;">
                    <div style="font-size:2.5rem;">{meta['emoji']}</div>
                    <div style="font-weight:700; color:{meta['color_hex']}; margin:0.5rem 0;">
                        {meta['label']}
                    </div>
                    <div style="font-size:0.72rem; color:#94A3B8;">
                        {meta['description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Footer ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center; color:#64748B; font-size:0.8rem; padding:1rem 0;'>
            Built with ❤️ using Python · TensorFlow · PyTorch · HuggingFace · Google Gemini<br>
            <a href='https://github.com/Harsh-maker007/Smartbridge_project1' 
               style='color:#6C63FF; text-decoration:none;'>
                📦 GitHub Repository
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
