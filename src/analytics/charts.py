"""
src/analytics/charts.py
=========================
Plotly chart generators for the Streamlit analytics dashboard.
All functions return a plotly Figure object ready for st.plotly_chart().
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from config.emotions import CHART_COLORS, EMOTION_META, ORDERED_LABELS


def emotion_pie_chart(emotion_counts: pd.Series) -> go.Figure:
    """
    Donut chart showing emotion frequency distribution.

    Args:
        emotion_counts: pd.Series with emotion labels as index, counts as values.

    Returns:
        Plotly Figure.
    """
    colors = [CHART_COLORS.get(e, "#888") for e in emotion_counts.index]
    fig = go.Figure(go.Pie(
        labels=[e.capitalize() for e in emotion_counts.index],
        values=emotion_counts.values,
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="#0F0F1A", width=2)),
        textinfo="label+percent",
        textfont_size=13,
    ))
    fig.update_layout(
        title=dict(text="Emotion Distribution", font=dict(size=16, color="#E2E8F0")),
        paper_bgcolor="#1A1A2E",
        plot_bgcolor="#1A1A2E",
        font=dict(color="#E2E8F0"),
        legend=dict(bgcolor="#1A1A2E", font=dict(color="#E2E8F0")),
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


def confidence_trend_chart(df: pd.DataFrame) -> go.Figure:
    """
    Line chart of confidence scores over time, colored by emotion.

    Args:
        df: Interaction log DataFrame with timestamp and final_confidence columns.

    Returns:
        Plotly Figure.
    """
    fig = px.line(
        df.sort_values("timestamp"),
        x="timestamp",
        y="final_confidence",
        color="final_emotion",
        color_discrete_map=CHART_COLORS,
        markers=True,
        title="Confidence Score Over Time",
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(
        paper_bgcolor="#1A1A2E",
        plot_bgcolor="#16213E",
        font=dict(color="#E2E8F0"),
        xaxis=dict(gridcolor="#2D3748", title="Time"),
        yaxis=dict(gridcolor="#2D3748", title="Confidence", range=[0, 1]),
        legend_title="Emotion",
        title_font=dict(size=16, color="#E2E8F0"),
        margin=dict(t=60, b=40, l=60, r=20),
    )
    return fig


def daily_activity_chart(daily_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of daily interaction counts.

    Args:
        daily_df: DataFrame with 'date' and 'count' columns.

    Returns:
        Plotly Figure.
    """
    fig = px.bar(
        daily_df,
        x="date",
        y="count",
        title="Daily Interactions",
        color_discrete_sequence=["#6C63FF"],
    )
    fig.update_layout(
        paper_bgcolor="#1A1A2E",
        plot_bgcolor="#16213E",
        font=dict(color="#E2E8F0"),
        xaxis=dict(gridcolor="#2D3748", title="Date"),
        yaxis=dict(gridcolor="#2D3748", title="Number of Sessions"),
        title_font=dict(size=16, color="#E2E8F0"),
        margin=dict(t=60, b=40, l=60, r=20),
    )
    return fig


def model_agreement_heatmap(crosstab: pd.DataFrame) -> go.Figure:
    """
    Heatmap showing BiLSTM vs BERT prediction agreement.

    Args:
        crosstab: pd.crosstab of bilstm_emotion vs bert_emotion.

    Returns:
        Plotly Figure.
    """
    emotions = [e.capitalize() for e in ORDERED_LABELS]
    z = crosstab.values if not crosstab.empty else np.zeros((5, 5))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=emotions,
        y=emotions,
        colorscale="Viridis",
        text=z,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title=dict(text="BiLSTM vs BERT Agreement Matrix", font=dict(size=16, color="#E2E8F0")),
        paper_bgcolor="#1A1A2E",
        plot_bgcolor="#1A1A2E",
        font=dict(color="#E2E8F0"),
        xaxis_title="BERT Prediction",
        yaxis_title="BiLSTM Prediction",
        margin=dict(t=60, b=60, l=80, r=20),
    )
    return fig


def confidence_bar_chart(probabilities: dict) -> go.Figure:
    """
    Horizontal bar chart for a single prediction's emotion probabilities.
    Used in the Streamlit home page.

    Args:
        probabilities: {emotion: probability} dict.

    Returns:
        Plotly Figure.
    """
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    emotions     = [e.capitalize() for e, _ in sorted_probs]
    probs        = [round(p * 100, 1) for _, p in sorted_probs]
    colors       = [CHART_COLORS.get(e.lower(), "#888") for e, _ in sorted_probs]

    fig = go.Figure(go.Bar(
        x=probs,
        y=emotions,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#0F0F1A", width=1)),
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(color="#E2E8F0", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        xaxis=dict(range=[0, 110], showgrid=False, showticklabels=False),
        yaxis=dict(gridcolor="#2D3748"),
        margin=dict(t=10, b=10, l=10, r=70),
        height=220,
        showlegend=False,
    )
    return fig
