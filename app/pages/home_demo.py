"""
app/pages/home_demo.py
=======================
Demo-mode home page — uses Gemini API for emotion detection
instead of BiLSTM/BERT models. Deployed on Streamlit Cloud.
Identical UX to full home.py but model-free.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import os

from config.emotions import EMOTION_META
from src.analytics.charts import confidence_bar_chart
from src.analytics.logger import log_interaction
from src.utils.helpers import generate_session_id
from src.utils.validators import validate_text_input


def render_home_demo():
    """Render the Demo-mode emotion detection home page."""

    # ── Check API Key ─────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        st.warning("🔑 **Gemini API Key Required**")
        st.markdown("Please enter your Gemini API key to continue. You can get one for free at [Google AI Studio](https://aistudio.google.com/).")
        user_key = st.text_input("Gemini API Key", type="password", placeholder="Paste your API key here...")
        if user_key:
            os.environ["GEMINI_API_KEY"] = user_key
            st.rerun()
        else:
            st.stop()

    # ── Page Header ───────────────────────────────────────────
    st.markdown(
        """
        <div style='text-align:center; padding: 1.5rem 0 1rem 0;'>
            <h1 style='font-size:2.2rem; font-weight:700; color:#E2E8F0; margin:0;'>
                🧠 Emotion-Aware Learning
            </h1>
            <p style='color:#94A3B8; font-size:1rem; margin-top:0.5rem;'>
                Describe your study situation · AI detects your emotion · Get personalized guidance
            </p>
            <div style='margin-top:0.75rem;'>
                <span style='background:rgba(108,99,255,0.15); color:#8B83FF; 
                             padding:3px 12px; border-radius:12px; font-size:0.78rem;
                             border:1px solid rgba(108,99,255,0.3);'>
                    ✨ Powered by Google Gemini AI + Rule-Based Engine
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Input ─────────────────────────────────────────────────
    col_inp, _ = st.columns([3, 1])
    with col_inp:
        st.markdown('<p class="section-header">📝 Describe Your Study Problem</p>', unsafe_allow_html=True)
        user_text = st.text_area(
            label="input",
            label_visibility="collapsed",
            placeholder=(
                "Example: I've been staring at this calculus problem for an hour "
                "and I still don't understand anything. I feel like giving up..."
            ),
            height=130,
            key="demo_input",
        )
        col_b1, col_b2, _ = st.columns([1.5, 1, 2])
        with col_b1:
            detect_clicked = st.button("🔍 Detect Emotion", use_container_width=True, type="primary")
        with col_b2:
            if st.button("🧹 Clear", use_container_width=True):
                st.session_state.pop("demo_result", None)
                st.session_state.pop("demo_guidance", None)
                st.rerun()

    # ── Run Detection ─────────────────────────────────────────
    if detect_clicked:
        is_valid, err = validate_text_input(user_text)
        if not is_valid:
            st.error(f"⚠️ {err}")
            return

        with st.spinner("🤖 Analyzing emotional state with Gemini AI..."):
            try:
                from src.detection.gemini_detector import get_gemini_detector
                detector = get_gemini_detector()
                result   = detector.detect(user_text)
                st.session_state["demo_result"] = result
            except Exception as e:
                st.error(f"❌ Detection failed: {e}")
                return

        with st.spinner("✨ Generating personalized learning guidance..."):
            try:
                from src.gemini.guidance_generator import generate_guidance
                guidance = generate_guidance(result)
                st.session_state["demo_guidance"] = guidance
            except Exception as e:
                guidance = {"guidance_text": "", "guidance_excerpt": "", "success": False}
                st.session_state["demo_guidance"] = guidance

        # Log to CSV
        try:
            log_interaction(
                detection_result=result,
                guidance_excerpt=guidance.get("guidance_excerpt", ""),
                session_id=st.session_state.get("session_id", generate_session_id()),
            )
        except Exception:
            pass  # Don't break on log failure in cloud

    # ── Results ───────────────────────────────────────────────
    if "demo_result" in st.session_state:
        result   = st.session_state["demo_result"]
        guidance = st.session_state.get("demo_guidance", {})

        emotion    = result["final_emotion"]
        confidence = result["final_confidence"]
        meta       = EMOTION_META.get(emotion, {})
        color      = meta.get("color_hex", "#6C63FF")
        emoji      = meta.get("emoji", "🔵")
        label      = meta.get("label", emotion.capitalize())

        st.markdown("---")

        # Row 1: Three columns
        col1, col2, col3 = st.columns([1.2, 1.8, 1.5])

        with col1:
            st.markdown('<p class="section-header">🎯 Detected Emotion</p>', unsafe_allow_html=True)

            if result.get("is_mixed"):
                st.markdown(
                    f'<div class="mixed-alert">⚡ Mixed: <strong>{result.get("mixed_label","")}</strong></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div class="emotion-card" style="border-color:{color}40; text-align:center;">
                    <span class="emotion-emoji">{emoji}</span>
                    <div class="emotion-label" style="color:{color};">{label}</div>
                    <br>
                    <span class="confidence-badge" style="border-color:{color};color:{color};">
                        {confidence:.0%} Confidence
                    </span>
                    <br><br>
                    <div style="font-size:0.78rem;color:#94A3B8;">
                        {meta.get('description','')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if result.get("rule_applied"):
                kws = [k for kl in result.get("keywords_matched", {}).values() for k in kl]
                tags = " ".join(f'<span class="keyword-tag">{k}</span>' for k in kws[:5])
                st.markdown(
                    f'<div style="margin-top:0.5rem;"><small style="color:#94A3B8;">🏷️ Keywords:</small><br>{tags}</div>',
                    unsafe_allow_html=True,
                )

        with col2:
            st.markdown('<p class="section-header">📊 Emotion Probabilities</p>', unsafe_allow_html=True)
            fig = confidence_bar_chart(result["ensemble_probabilities"])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col3:
            st.markdown('<p class="section-header">🔍 Detection Details</p>', unsafe_allow_html=True)
            gap = result.get("emotion_gap", 1.0)
            certainty = "🟢 High" if confidence > 0.65 else ("🟡 Moderate" if confidence > 0.45 else "🔴 Low")
            mixed_txt = f"⚡ {result.get('mixed_label','')}" if result.get("is_mixed") else "✅ Single Emotion"

            st.markdown(
                f"""
                <div class="emotion-card">
                    <div style="margin-bottom:0.75rem;">
                        <div style="font-size:0.8rem;color:#94A3B8;">Detection Mode</div>
                        <div style="color:#8B83FF;font-weight:600;">✨ Gemini AI + Rules</div>
                    </div>
                    <div style="margin-bottom:0.75rem;">
                        <div style="font-size:0.8rem;color:#94A3B8;">Certainty</div>
                        <div style="font-weight:600;">{certainty} ({confidence:.0%})</div>
                    </div>
                    <div style="margin-bottom:0.75rem;">
                        <div style="font-size:0.8rem;color:#94A3B8;">Emotion State</div>
                        <div style="font-weight:600;">{mixed_txt}</div>
                    </div>
                    <div>
                        <div style="font-size:0.8rem;color:#94A3B8;">Top-2 Gap</div>
                        <div style="font-weight:600;">{gap:.1%}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Row 2: Gemini Guidance
        if guidance.get("guidance_text"):
            st.markdown("---")
            st.markdown('<p class="section-header">🤖 Personalized Learning Guidance</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="guidance-panel">{guidance["guidance_text"]}</div>',
                unsafe_allow_html=True,
            )

        # Row 3: History
        with st.expander("📋 Session History", expanded=False):
            from src.analytics.aggregator import load_logs
            df = load_logs()
            if df is None or df.empty:
                st.info("No history yet — make your first detection above!")
            else:
                disp = df[["timestamp", "final_emotion", "final_confidence", "is_mixed"]].tail(10)
                disp.columns = ["Timestamp", "Emotion", "Confidence", "Mixed"]
                disp["Confidence"] = disp["Confidence"].apply(lambda x: f"{float(x):.0%}")
                st.dataframe(disp, use_container_width=True, hide_index=True)
