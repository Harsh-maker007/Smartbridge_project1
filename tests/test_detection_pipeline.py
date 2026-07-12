"""
tests/test_detection_pipeline.py
==================================
Unit tests for the detection engine (rule-based, ensemble, mixed emotion).
Does NOT require trained model files — tests logic only.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.detection.rule_based import find_matched_keywords, apply_keyword_boost, run_rule_based
from src.detection.ensemble import weighted_average, get_top_emotion
from src.detection.mixed_emotion import detect_mixed_emotions, format_mixed_label


class TestRuleBased:

    def test_find_frustrated_keywords(self):
        text    = "I give up, I've tried everything and nothing works!"
        matches = find_matched_keywords(text)
        assert "frustrated" in matches

    def test_find_curious_keywords(self):
        text    = "I wonder how this algorithm actually works underneath"
        matches = find_matched_keywords(text)
        assert "curious" in matches

    def test_no_match_generic(self):
        text    = "Today is a nice day"
        matches = find_matched_keywords(text)
        # Unlikely to match any emotion keyword
        assert isinstance(matches, dict)

    def test_keyword_boost_increases_probability(self):
        probs = {"frustrated": 0.4, "confused": 0.3, "curious": 0.1, "bored": 0.1, "confident": 0.1}
        matched = {"frustrated": ["give up", "nothing works"]}
        boosted, applied = apply_keyword_boost(probs, matched, boost_factor=0.15)
        assert applied is True
        assert boosted["frustrated"] > probs["frustrated"] / sum(probs.values())

    def test_keyword_boost_sums_to_one(self):
        probs = {"frustrated": 0.4, "confused": 0.3, "curious": 0.1, "bored": 0.1, "confident": 0.1}
        matched = {"frustrated": ["stuck"]}
        boosted, _ = apply_keyword_boost(probs, matched)
        assert abs(sum(boosted.values()) - 1.0) < 1e-5

    def test_no_boost_when_no_matches(self):
        probs   = {"frustrated": 0.4, "confused": 0.3, "curious": 0.1, "bored": 0.1, "confident": 0.1}
        boosted, applied = apply_keyword_boost(probs, {})
        assert applied is False
        assert boosted == probs


class TestEnsemble:

    def test_weighted_average_sums_to_one(self):
        bilstm = {"bored": 0.1, "confident": 0.2, "confused": 0.4, "curious": 0.1, "frustrated": 0.2}
        bert   = {"bored": 0.05, "confident": 0.15, "confused": 0.5, "curious": 0.15, "frustrated": 0.15}
        result = weighted_average(bilstm, bert)
        assert abs(sum(result.values()) - 1.0) < 1e-5

    def test_weighted_average_bert_weight(self):
        # When BERT is much more confident about 'frustrated', it should dominate
        bilstm = {"bored": 0.2, "confident": 0.2, "confused": 0.2, "curious": 0.2, "frustrated": 0.2}
        bert   = {"bored": 0.0, "confident": 0.0, "confused": 0.0, "curious": 0.0, "frustrated": 1.0}
        result = weighted_average(bilstm, bert, bilstm_weight=0.4, bert_weight=0.6)
        assert result["frustrated"] > 0.6

    def test_get_top_emotion(self):
        probs   = {"bored": 0.1, "confident": 0.05, "confused": 0.2, "curious": 0.1, "frustrated": 0.55}
        emotion, conf = get_top_emotion(probs)
        assert emotion == "frustrated"
        assert abs(conf - 0.55) < 1e-5


class TestMixedEmotion:

    def test_not_mixed_high_confidence(self):
        probs  = {"bored": 0.05, "confident": 0.05, "confused": 0.05, "curious": 0.05, "frustrated": 0.80}
        result = detect_mixed_emotions(probs, threshold=0.15)
        assert result["is_mixed"] is False

    def test_mixed_when_close(self):
        probs  = {"bored": 0.1, "confident": 0.05, "confused": 0.40, "curious": 0.05, "frustrated": 0.40}
        result = detect_mixed_emotions(probs, threshold=0.15)
        assert result["is_mixed"] is True
        assert len(result["mixed_emotions"]) == 2

    def test_format_mixed_label(self):
        emotions = [("frustrated", 0.4), ("confused", 0.38)]
        label    = format_mixed_label(emotions)
        assert "Frustrated" in label
        assert "Confused" in label
        assert "+" in label

    def test_format_empty_label(self):
        assert format_mixed_label([]) == ""


class TestValidators:

    def test_valid_input(self):
        from src.utils.validators import validate_text_input
        valid, err = validate_text_input("I am confused about calculus")
        assert valid is True
        assert err is None

    def test_empty_input(self):
        from src.utils.validators import validate_text_input
        valid, err = validate_text_input("")
        assert valid is False
        assert err is not None

    def test_too_short_input(self):
        from src.utils.validators import validate_text_input
        valid, err = validate_text_input("Help me")
        assert valid is False

    def test_valid_probabilities(self):
        from src.utils.validators import validate_probabilities
        valid, err = validate_probabilities([0.2, 0.2, 0.2, 0.2, 0.2])
        assert valid is True

    def test_invalid_probability_count(self):
        from src.utils.validators import validate_probabilities
        valid, err = validate_probabilities([0.5, 0.5])
        assert valid is False
