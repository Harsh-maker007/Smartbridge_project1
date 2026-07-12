"""
src/preprocessing/tokenizer.py
================================
Tokenization utilities for both BiLSTM (Keras word-index tokenizer)
and BERT (HuggingFace AutoTokenizer).
Handles fitting, transforming, padding, and persistence.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer

from config.settings import (
    TOKENIZER_PATH,
    BILSTM_MAX_VOCAB_SIZE,
    BILSTM_MAX_SEQUENCE_LEN,
    BERT_BASE_MODEL,
    BERT_MAX_LENGTH,
)
from src.utils.helpers import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# BiLSTM Tokenizer (Keras)
# ═══════════════════════════════════════════════════════════════

class BiLSTMTokenizer:
    """
    Wrapper around Keras Tokenizer for the BiLSTM model.

    Workflow:
        1. fit(texts)     — learn vocabulary from training corpus
        2. transform(texts) — convert texts to padded integer sequences
        3. save() / load() — persist to disk
    """

    def __init__(
        self,
        num_words: int = BILSTM_MAX_VOCAB_SIZE,
        max_length: int = BILSTM_MAX_SEQUENCE_LEN,
        oov_token: str = "<OOV>",
    ):
        self.num_words  = num_words
        self.max_length = max_length
        self.oov_token  = oov_token
        self._tokenizer = Tokenizer(
            num_words=num_words,
            oov_token=oov_token,
            lower=True,
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> "BiLSTMTokenizer":
        """
        Fit tokenizer on a list of cleaned text strings.

        Args:
            texts: List of pre-cleaned text samples.

        Returns:
            self (for chaining)
        """
        logger.info(f"Fitting BiLSTM tokenizer on {len(texts)} samples...")
        self._tokenizer.fit_on_texts(texts)
        self._fitted = True
        vocab_size = min(self.num_words, len(self._tokenizer.word_index) + 1)
        logger.info(f"Vocabulary size: {vocab_size:,} tokens")
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to padded integer sequences.

        Args:
            texts: List of cleaned text strings.

        Returns:
            np.ndarray of shape (n_samples, max_length).
        """
        self._check_fitted()
        sequences = self._tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding="post",
            truncating="post",
        )
        return padded

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit then transform in one step."""
        return self.fit(texts).transform(texts)

    @property
    def vocab_size(self) -> int:
        """Return actual vocabulary size (capped at num_words)."""
        self._check_fitted()
        return min(self.num_words, len(self._tokenizer.word_index) + 1)

    @property
    def word_index(self) -> Dict[str, int]:
        """Return full word→index mapping."""
        self._check_fitted()
        return self._tokenizer.word_index

    def save(self, path: Path = TOKENIZER_PATH) -> None:
        """Persist tokenizer to disk."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"BiLSTM Tokenizer saved to: {path}")

    @classmethod
    def load(cls, path: Path = TOKENIZER_PATH) -> "BiLSTMTokenizer":
        """Load a previously saved tokenizer from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BiLSTM Tokenizer not found at: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"BiLSTM Tokenizer loaded from: {path}")
        return obj

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Tokenizer not fitted. Call .fit() first.")


# ═══════════════════════════════════════════════════════════════
# BERT Tokenizer (HuggingFace)
# ═══════════════════════════════════════════════════════════════

class BERTTokenizer:
    """
    Thin wrapper around HuggingFace AutoTokenizer for bert-base-uncased.

    Returns PyTorch-ready tensors (input_ids, attention_mask).
    """

    def __init__(
        self,
        model_name: str = BERT_BASE_MODEL,
        max_length: int = BERT_MAX_LENGTH,
    ):
        self.model_name = model_name
        self.max_length = max_length
        logger.info(f"Loading HuggingFace tokenizer: {model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, texts: List[str]) -> Dict[str, Any]:
        """
        Tokenize a list of texts for BERT.

        Args:
            texts: List of text strings (lightly cleaned).

        Returns:
            Dict with keys: input_ids, attention_mask, token_type_ids
            Each value is a tensor of shape (n_samples, max_length).
        """
        encoding = self._tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",      # PyTorch tensors
            return_attention_mask=True,
        )
        return encoding

    def tokenize_single(self, text: str) -> Dict[str, Any]:
        """Tokenize a single text string for inference."""
        return self.tokenize([text])

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size
