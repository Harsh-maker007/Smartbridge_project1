"""
app/pages/home.py
==================
Home page — student emotion detection with Gemini guidance.
The primary user-facing page of the application.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from src.detection.detector import get_detector
from src.gemini.guidance_generator import generate_guidance
from src.analytics.logger import log_interaction
from src.analytics.charts import confidence_bar_chart
from config.emotions import EMOTION_META, CHART_COLORS
from src.utils.validators import validate_text_input
from src.utils.helpers import generate_session_id


def render_home():
    """Render the emotion detection home page."""

    # ── Page Header ───────────────────────────────────────────
    st.markdown(
        """
        <div style='text-align:center; padding: 1.5rem 0 1rem 0;'>
            <h1 style='font-size:2.2rem; font-weight:700; color:#E2E8F0; margin:0;'>
                🧠 Emotion-Aware Learning
            </h1>
            <p style='color:#94A3B8; font-size:1rem; margin-top:0.5rem;'>
                Describe your study situation and get personalized AI-powered guidance
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Input Section ─────────────────────────────────────────
    col_input, col_spacer = st.columns([3, 1])

    with col_input:
        st.markdown('<p class="section-header">📝 Describe Your Study Problem</p>', unsafe_allow_html=True)
        user_text = st.text_area(
            label="student_input",
            label_visibility="collapsed",
            placeholder=(
                "Example: I've been staring at this calculus problem for an hour "
                "and I still don't get it. I feel like giving up..."
            ),
            height=130,
            key="student_input",
        )

        col_btn1, col_btn2, _ = st.columns([1.5, 1, 2])
        with col_btn1:
            detect_clicked = st.button("🔍 Detect Emotion", use_container_width=True, type="primary")
        with col_btn2:
            if st.button("🧹 Clear", use_container_width=True):
                st.session_state.pop("last_result", None)
                st.rerun()

    # ── Detect on button click ────────────────────────────────
    if detect_clicked:
        is_valid, error_msg = validate_text_input(user_text)

        if not is_valid:
            st.error(f"⚠️ {error_msg}")
            return

        with st.spinner("🤖 Analyzing emotional state..."):
            try:
                detector = get_detector()
                result   = detector.detect(user_text)
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"❌ Detection failed: {e}")
                return

        with st.spinner("✨ Generating personalized guidance..."):
            try:
                guidance = generate_guidance(result)
                st.session_state["last_guidance"] = guidance
            except Exception as e:
                guidance = {"guidance_text": "", "guidance_excerpt": "", "success": False}

        # Log to CSV
        log_interaction(
            detection_result=result,
            guidance_excerpt=guidance.get("guidance_excerpt", ""),
            session_id=st.session_state.get("session_id", generate_session_id()),
        )

    # ── Results Section ───────────────────────────────────────
    if "last_result" in st.session_state:
        result   = st.session_state["last_result"]
        guidance = st.session_state.get("last_guidance", {})

        emotion      = result["final_emotion"]
        confidence   = result["final_confidence"]
        meta         = EMOTION_META.get(emotion, {})
        color        = meta.get("color_hex", "#6C63FF")
        emoji        = meta.get("emoji", "🔵")
        label        = meta.get("label", emotion.capitalize())

        st.markdown("---")

        # Row 1: Emotion Card | Confidence Bars | Model Comparison
        col1, col2, col3 = st.columns([1.2, 1.8, 1.5])

        with col1:
            _render_emotion_card(emotion, label, emoji, confidence, color, result)

        with col2:
            st.markdown('<p class="section-header">📊 Emotion Probabilities</p>', unsafe_allow_html=True)
            probs = result["ensemble_probabilities"]
            fig = confidence_bar_chart(probs)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col3:
            _render_model_comparison(result)

        # Row 2: Gemini Guidance (full width)
        if guidance.get("guidance_text"):
            st.markdown("---")
            _render_guidance(guidance["guidance_text"])

        # Row 3: Interaction History (expandable)
        with st.expander("📋 Session History", expanded=False):
            _render_session_history()


def _render_emotion_card(emotion, label, emoji, confidence, color, result):
    """Render the primary emotion result card."""
    st.markdown('<p class="section-header">🎯 Detected Emotion</p>', unsafe_allow_html=True)

    # Mixed emotion alert
    if result.get("is_mixed"):
        st.markdown(
            f'<div class="mixed-alert">⚡ Mixed Emotions: <strong>{result.get("mixed_label", "")}</strong></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="emotion-card" style="border-color: {color}40; text-align: center;">
            <span class="emotion-emoji">{emoji}</span>
            <div class="emotion-label" style="color: {color};">{label}</div>
            <br>
            <span class="confidence-badge" style="border-color:{color}; color:{color};">
                {confidence:.0%} Confidence
            </span>
            <br><br>
            <div style="font-size:0.78rem; color:#94A3B8; margin-top:0.5rem;">
                {meta_description(emotion)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Rule applied indicator
    if result.get("rule_applied"):
        keywords = result.get("keywords_matched", {})
        kw_flat  = [kw for kws in keywords.values() for kw in kws]
        kw_tags  = " ".join(f'<span class="keyword-tag">{k}</span>' for k in kw_flat[:5])
        st.markdown(
            f'<div style="margin-top:0.5rem;"><small style="color:#94A3B8;">🏷️ Keywords:</small><br>{kw_tags}</div>',
            unsafe_allow_html=True,
        )


def _render_model_comparison(result):
    """Render BiLSTM vs BERT comparison panel."""
    st.markdown('<p class="section-header">⚖️ Model Comparison</p>', unsafe_allow_html=True)

    bilstm = result.get("bilstm", {})
    bert   = result.get("bert", {})
    agree  = result.get("models_agree", False)

    agree_badge = (
        '<span style="color:#10B981;">✅ Models Agree</span>'
        if agree else
        '<span style="color:#F59E0B;">⚡ Models Differ</span>'
    )

    st.markdown(
        f"""
        <div class="emotion-card">
            <div style="text-align:center; margin-bottom:1rem;">{agree_badge}</div>
            <div class="model-badge model-badge-bilstm" style="margin-bottom:0.5rem; width:100%;">
                🔵 BiLSTM: {bilstm.get('emotion','').capitalize()}
                <span style="margin-left:auto; font-weight:700;">{bilstm.get('confidence',0):.0%}</span>
            </div>
            <div class="model-badge model-badge-bert" style="width:100%;">
                🟢 BERT: {bert.get('emotion','').capitalize()}
                <span style="margin-left:auto; font-weight:700;">{bert.get('confidence',0):.0%}</span>
            </div>
            <div style="margin-top:1rem; font-size:0.78rem; color:#94A3B8; text-align:center;">
                Ensemble: 40% BiLSTM + 60% BERT
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_guidance(guidance_text: str):
    """Render the Gemini guidance panel."""
    st.markdown('<p class="section-header">🤖 Personalized Learning Guidance</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="guidance-panel">{guidance_text}</div>',
        unsafe_allow_html=True,
    )


def _render_session_history():
    """Render recent session history from CSV logs."""
    from src.analytics.aggregator import load_logs
    df = load_logs()
    if df is None or df.empty:
        st.info("No interaction history yet. Make your first detection above!")
        return

    display = df[["timestamp", "final_emotion", "final_confidence", "is_mixed"]].tail(10)
    display.columns = ["Timestamp", "Emotion", "Confidence", "Mixed"]
    display["Confidence"] = display["Confidence"].apply(lambda x: f"{float(x):.0%}")
    st.dataframe(display, use_container_width=True, hide_index=True)


def meta_description(emotion: str) -> str:
    from config.emotions import EMOTION_META
    return EMOTION_META.get(emotion, {}).get("description", "")
