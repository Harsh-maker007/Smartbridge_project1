"""
src/analytics/aggregator.py
==============================
Reads interaction_logs.csv and computes aggregated analytics stats.
Used by the Streamlit analytics dashboard.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from config.settings import INTERACTION_LOG_PATH
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def load_logs() -> Optional[pd.DataFrame]:
    """
    Load interaction logs from CSV.

    Returns:
        DataFrame or None if no logs exist yet.
    """
    if not INTERACTION_LOG_PATH.exists():
        logger.info("No interaction logs found.")
        return None

    df = pd.read_csv(INTERACTION_LOG_PATH, parse_dates=["timestamp"])

    if df.empty:
        return None

    # Ensure correct types
    df["final_confidence"]  = pd.to_numeric(df["final_confidence"], errors="coerce")
    df["bilstm_confidence"] = pd.to_numeric(df["bilstm_confidence"], errors="coerce")
    df["bert_confidence"]   = pd.to_numeric(df["bert_confidence"], errors="coerce")
    df["is_mixed"]          = df["is_mixed"].astype(str).str.lower() == "true"

    logger.info(f"Loaded {len(df)} interaction logs.")
    return df


def get_emotion_distribution(df: pd.DataFrame) -> pd.Series:
    """Emotion frequency counts."""
    return df["final_emotion"].value_counts()


def get_avg_confidence_per_emotion(df: pd.DataFrame) -> pd.DataFrame:
    """Average confidence score per emotion."""
    return df.groupby("final_emotion")["final_confidence"].mean().reset_index()


def get_daily_interaction_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Number of interactions per day."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    return df.groupby("date").size().reset_index(name="count")


def get_model_agreement_rate(df: pd.DataFrame) -> float:
    """Fraction of sessions where BiLSTM and BERT agreed."""
    if df.empty or "bilstm_emotion" not in df.columns:
        return 0.0
    agree = (df["bilstm_emotion"] == df["bert_emotion"]).sum()
    return float(agree) / len(df)


def get_mixed_emotion_rate(df: pd.DataFrame) -> float:
    """Fraction of sessions flagged as mixed emotions."""
    if df.empty:
        return 0.0
    return float(df["is_mixed"].sum()) / len(df)


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Compute all top-level summary statistics.

    Returns:
        Dict with total_sessions, most_common_emotion, avg_confidence,
        model_agreement_rate, mixed_emotion_rate.
    """
    if df is None or df.empty:
        return {
            "total_sessions":      0,
            "most_common_emotion": "N/A",
            "avg_confidence":      0.0,
            "model_agreement_rate": 0.0,
            "mixed_emotion_rate":  0.0,
        }

    return {
        "total_sessions":      len(df),
        "most_common_emotion": df["final_emotion"].mode().iloc[0] if not df.empty else "N/A",
        "avg_confidence":      float(df["final_confidence"].mean()),
        "model_agreement_rate": get_model_agreement_rate(df),
        "mixed_emotion_rate":  get_mixed_emotion_rate(df),
    }


def get_bilstm_vs_bert_confusion(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulation of BiLSTM vs BERT predictions (agreement matrix)."""
    if df.empty:
        return pd.DataFrame()
    return pd.crosstab(
        df["bilstm_emotion"],
        df["bert_emotion"],
        rownames=["BiLSTM"],
        colnames=["BERT"],
    )
