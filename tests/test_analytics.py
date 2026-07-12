"""
tests/test_analytics.py
=========================
Unit tests for analytics logger, aggregator, and chart generators.
"""

import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd


FAKE_RESULT = {
    "timestamp":          "2024-01-15 10:30:00",
    "input_text":         "I am totally confused about derivatives",
    "bilstm":             {"emotion": "confused", "confidence": 0.72},
    "bert":               {"emotion": "confused", "confidence": 0.85},
    "final_emotion":      "confused",
    "final_confidence":   0.80,
    "is_mixed":           False,
    "mixed_emotions":     [],
    "mixed_label":        "",
    "rule_applied":       True,
    "keywords_matched":   {"confused": ["confused"]},
}


class TestLogger:

    def test_log_creates_file(self, tmp_path):
        # Override log path temporarily
        import config.settings as s
        original = s.INTERACTION_LOG_PATH
        s.INTERACTION_LOG_PATH = tmp_path / "test_logs.csv"

        from src.analytics.logger import log_interaction
        sid = log_interaction(FAKE_RESULT, guidance_excerpt="Study tip here.")

        assert (tmp_path / "test_logs.csv").exists()
        assert isinstance(sid, str)

        # Restore
        s.INTERACTION_LOG_PATH = original

    def test_log_row_content(self, tmp_path):
        import config.settings as s
        log_path = tmp_path / "test_logs2.csv"
        s.INTERACTION_LOG_PATH = log_path

        from src.analytics.logger import log_interaction
        log_interaction(FAKE_RESULT, guidance_excerpt="Be patient.")
        df = pd.read_csv(log_path)

        assert len(df) == 1
        assert df.iloc[0]["final_emotion"] == "confused"
        assert float(df.iloc[0]["final_confidence"]) == pytest.approx(0.80, abs=0.01)

        s.INTERACTION_LOG_PATH = Path("data/interactions/interaction_logs.csv")


class TestAggregator:

    @pytest.fixture
    def sample_df(self):
        """Create a sample interaction log DataFrame."""
        return pd.DataFrame([
            {"timestamp": "2024-01-10", "final_emotion": "frustrated", "final_confidence": 0.82,
             "bilstm_emotion": "frustrated", "bert_emotion": "frustrated", "is_mixed": False},
            {"timestamp": "2024-01-10", "final_emotion": "confused",   "final_confidence": 0.70,
             "bilstm_emotion": "confused",   "bert_emotion": "confused",   "is_mixed": False},
            {"timestamp": "2024-01-11", "final_emotion": "frustrated", "final_confidence": 0.65,
             "bilstm_emotion": "confused",   "bert_emotion": "frustrated", "is_mixed": True},
            {"timestamp": "2024-01-11", "final_emotion": "curious",    "final_confidence": 0.55,
             "bilstm_emotion": "curious",    "bert_emotion": "curious",    "is_mixed": False},
        ])

    def test_emotion_distribution(self, sample_df):
        from src.analytics.aggregator import get_emotion_distribution
        dist = get_emotion_distribution(sample_df)
        assert dist["frustrated"] == 2
        assert dist["confused"] == 1

    def test_model_agreement_rate(self, sample_df):
        from src.analytics.aggregator import get_model_agreement_rate
        rate = get_model_agreement_rate(sample_df)
        assert 0.0 <= rate <= 1.0
        # 3 out of 4 agree
        assert abs(rate - 0.75) < 0.01

    def test_mixed_emotion_rate(self, sample_df):
        from src.analytics.aggregator import get_mixed_emotion_rate
        rate = get_mixed_emotion_rate(sample_df)
        assert abs(rate - 0.25) < 0.01  # 1 out of 4 is mixed

    def test_summary_stats(self, sample_df):
        from src.analytics.aggregator import get_summary_stats
        stats = get_summary_stats(sample_df)
        assert stats["total_sessions"] == 4
        assert stats["most_common_emotion"] == "frustrated"
        assert 0.0 <= stats["avg_confidence"] <= 1.0


class TestCharts:

    def test_confidence_bar_chart(self):
        from src.analytics.charts import confidence_bar_chart
        probs = {"bored": 0.05, "confident": 0.10, "confused": 0.15, "curious": 0.20, "frustrated": 0.50}
        fig = confidence_bar_chart(probs)
        assert fig is not None

    def test_emotion_pie_chart(self):
        from src.analytics.charts import emotion_pie_chart
        dist = pd.Series({"frustrated": 5, "confused": 3, "curious": 2})
        fig  = emotion_pie_chart(dist)
        assert fig is not None
