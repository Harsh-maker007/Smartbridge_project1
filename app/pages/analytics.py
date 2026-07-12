"""
app/pages/analytics.py
========================
Analytics dashboard — emotion trends, session statistics, model agreement,
and interaction history with export functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
from src.analytics.aggregator import (
    load_logs, get_emotion_distribution, get_avg_confidence_per_emotion,
    get_daily_interaction_counts, get_model_agreement_rate,
    get_mixed_emotion_rate, get_summary_stats, get_bilstm_vs_bert_confusion,
)
from src.analytics.charts import (
    emotion_pie_chart, confidence_trend_chart,
    daily_activity_chart, model_agreement_heatmap,
)
from config.emotions import EMOTION_META


def render_analytics():
    """Render the analytics dashboard page."""

    st.markdown(
        """
        <div style='padding: 1rem 0;'>
            <h1 style='font-size:2rem; font-weight:700; color:#E2E8F0; margin:0;'>
                📊 Analytics Dashboard
            </h1>
            <p style='color:#94A3B8; margin-top:0.5rem;'>
                Session insights, emotion trends, and model performance
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load Data ─────────────────────────────────────────────
    df = load_logs()

    if df is None or df.empty:
        st.markdown(
            """
            <div class="emotion-card" style="text-align:center; padding:3rem;">
                <div style="font-size:3rem;">📭</div>
                <h3 style="color:#94A3B8;">No Data Yet</h3>
                <p style="color:#64748B;">Make some detections on the Home page to see analytics here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    stats = get_summary_stats(df)

    # ── Summary Metrics Row ───────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics_data = [
        (m1, str(stats["total_sessions"]),              "Total Sessions",       "🔢"),
        (m2, stats["most_common_emotion"].capitalize(),  "Most Common Emotion",  "🏆"),
        (m3, f"{stats['avg_confidence']:.0%}",           "Avg Confidence",       "🎯"),
        (m4, f"{stats['model_agreement_rate']:.0%}",     "Model Agreement",      "🤝"),
        (m5, f"{stats['mixed_emotion_rate']:.0%}",       "Mixed Emotions",       "⚡"),
    ]
    for col, value, label, icon in metrics_data:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Charts Row 1 ──────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">🥧 Emotion Distribution</p>', unsafe_allow_html=True)
        emotion_dist = get_emotion_distribution(df)
        if not emotion_dist.empty:
            fig = emotion_pie_chart(emotion_dist)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown('<p class="section-header">📈 Daily Activity</p>', unsafe_allow_html=True)
        daily_df = get_daily_interaction_counts(df)
        if not daily_df.empty:
            fig = daily_activity_chart(daily_df)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Charts Row 2 ──────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<p class="section-header">📉 Confidence Over Time</p>', unsafe_allow_html=True)
        fig = confidence_trend_chart(df)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col4:
        st.markdown('<p class="section-header">🔥 Model Agreement Matrix</p>', unsafe_allow_html=True)
        crosstab = get_bilstm_vs_bert_confusion(df)
        fig = model_agreement_heatmap(crosstab)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Interaction History Table ─────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">📋 Interaction History</p>', unsafe_allow_html=True)

    col_filter1, col_filter2, col_export = st.columns([1.5, 1.5, 1])
    with col_filter1:
        emotion_filter = st.multiselect(
            "Filter by Emotion",
            options=df["final_emotion"].unique().tolist(),
            default=[],
        )
    with col_filter2:
        model_filter = st.selectbox(
            "Filter by Model Agreement",
            options=["All", "Agreed", "Disagreed"],
        )
    with col_export:
        st.markdown("<br>", unsafe_allow_html=True)
        csv_data = df.to_csv(index=False)
        st.download_button(
            "⬇️ Export CSV",
            data=csv_data,
            file_name="interaction_logs.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Apply filters
    filtered_df = df.copy()
    if emotion_filter:
        filtered_df = filtered_df[filtered_df["final_emotion"].isin(emotion_filter)]
    if model_filter == "Agreed":
        filtered_df = filtered_df[filtered_df["bilstm_emotion"] == filtered_df["bert_emotion"]]
    elif model_filter == "Disagreed":
        filtered_df = filtered_df[filtered_df["bilstm_emotion"] != filtered_df["bert_emotion"]]

    display_cols = ["timestamp", "student_input", "final_emotion", "final_confidence",
                    "bilstm_emotion", "bert_emotion", "is_mixed"]
    display_df = filtered_df[display_cols].tail(50).sort_values("timestamp", ascending=False)
    display_df.columns = ["Time", "Input", "Emotion", "Confidence", "BiLSTM", "BERT", "Mixed"]
    display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{float(x):.0%}")
    display_df["Input"] = display_df["Input"].apply(lambda x: str(x)[:80] + "..." if len(str(x)) > 80 else str(x))

    st.dataframe(display_df, use_container_width=True, hide_index=True)
