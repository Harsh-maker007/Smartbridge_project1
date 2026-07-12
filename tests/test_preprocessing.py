"""
tests/test_preprocessing.py
==============================
Unit tests for the preprocessing pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.preprocessing.cleaner import (
    clean_text, clean_for_bilstm, clean_for_bert,
    remove_html, expand_contractions, remove_urls,
)


class TestCleaner:

    def test_remove_html(self):
        assert remove_html("<b>Hello</b> World") == " Hello  World"

    def test_expand_contractions(self):
        result = expand_contractions("I can't understand this")
        assert "cannot" in result or "can not" in result

    def test_remove_urls(self):
        result = remove_urls("Visit https://example.com for help")
        assert "https" not in result

    def test_clean_text_basic(self):
        result = clean_text("I don't understand this AT ALL!!!")
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == result.lower()

    def test_clean_text_empty(self):
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_clean_for_bert_preserves_more(self):
        text = "I can't solve this problem! It's really hard."
        bilstm_result = clean_for_bilstm(text)
        bert_result   = clean_for_bert(text)
        # BERT version should be longer (less aggressive cleaning)
        assert len(bert_result) >= len(bilstm_result)

    def test_clean_for_bilstm_lowercase(self):
        result = clean_for_bilstm("FRUSTRATED And CONFUSED")
        assert result == result.lower()


class TestEncoder:

    def test_build_default_encoder(self):
        from src.preprocessing.encoder import build_default_encoder
        enc = build_default_encoder()
        assert enc.num_classes == 5
        assert "frustrated" in enc.classes

    def test_transform_inverse(self):
        from src.preprocessing.encoder import build_default_encoder
        enc = build_default_encoder()
        labels   = ["bored", "confident", "confused"]
        indices  = enc.transform(labels)
        restored = enc.inverse_transform(indices)
        assert list(restored) == labels

    def test_onehot_shape(self):
        from src.preprocessing.encoder import build_default_encoder
        import numpy as np
        enc = build_default_encoder()
        oh  = enc.to_onehot(["bored", "frustrated"])
        assert oh.shape == (2, 5)
        assert oh.sum(axis=1).tolist() == [1.0, 1.0]
