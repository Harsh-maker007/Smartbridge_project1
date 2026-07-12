"""
src/models/bilstm/inference.py
================================
BiLSTM model inference — loads saved model + tokenizer and
returns emotion probabilities for a given input text.
"""

import numpy as np
import tensorflow as tf
from typing import Optional

from config.settings import BILSTM_MODEL_PATH, EMOTION_LABELS
from src.preprocessing.cleaner import clean_for_bilstm
from src.preprocessing.tokenizer import BiLSTMTokenizer
from src.preprocessing.encoder import EmotionLabelEncoder, build_default_encoder
from src.utils.helpers import get_logger, softmax_to_dict

logger = get_logger(__name__)


class BiLSTMInference:
    """
    Inference engine for the trained BiLSTM emotion classifier.

    Usage:
        inferencer = BiLSTMInference()
        inferencer.load()
        result = inferencer.predict("I can't understand this at all")
        # result = {"emotion": "confused", "confidence": 0.87, "probabilities": {...}}
    """

    def __init__(self):
        self.model:     Optional[tf.keras.Model]   = None
        self.tokenizer: Optional[BiLSTMTokenizer]  = None
        self.encoder:   Optional[EmotionLabelEncoder] = None
        self._loaded = False

    def load(self) -> "BiLSTMInference":
        """Load model, tokenizer, and label encoder from disk."""
        if self._loaded:
            return self

        if not BILSTM_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"BiLSTM model not found at {BILSTM_MODEL_PATH}.\n"
                f"Run: python scripts/train_bilstm.py"
            )

        logger.info(f"Loading BiLSTM model from: {BILSTM_MODEL_PATH}")
        self.model     = tf.keras.models.load_model(str(BILSTM_MODEL_PATH))
        self.tokenizer = BiLSTMTokenizer.load()
        self.encoder   = build_default_encoder()
        self._loaded   = True
        logger.info("BiLSTM model loaded successfully.")
        return self

    def predict(self, text: str) -> dict:
        """
        Predict emotion for a single text input.

        Args:
            text: Raw student input text.

        Returns:
            Dict with:
                - emotion:       str  — top predicted emotion
                - confidence:    float — probability of top emotion
                - probabilities: dict  — {emotion: probability} for all 5 classes
        """
        if not self._loaded:
            self.load()

        # Clean and tokenize
        cleaned   = clean_for_bilstm(text)
        sequences = self.tokenizer.transform([cleaned])      # shape (1, max_len)

        # Predict
        probs = self.model.predict(sequences, verbose=0)[0]  # shape (5,)

        # Map to labels
        labels     = sorted(EMOTION_LABELS)
        prob_dict  = softmax_to_dict(probs.tolist(), labels)
        top_idx    = int(np.argmax(probs))
        top_emotion = labels[top_idx]
        confidence  = float(probs[top_idx])

        return {
            "emotion":       top_emotion,
            "confidence":    confidence,
            "probabilities": prob_dict,
            "model":         "bilstm",
        }

    def predict_batch(self, texts: list) -> list:
        """
        Predict emotions for a batch of texts.

        Args:
            texts: List of raw text strings.

        Returns:
            List of prediction dicts.
        """
        if not self._loaded:
            self.load()

        cleaned   = [clean_for_bilstm(t) for t in texts]
        sequences = self.tokenizer.transform(cleaned)
        probs_all = self.model.predict(sequences, batch_size=64, verbose=0)

        labels = sorted(EMOTION_LABELS)
        results = []
        for probs in probs_all:
            top_idx = int(np.argmax(probs))
            results.append({
                "emotion":       labels[top_idx],
                "confidence":    float(probs[top_idx]),
                "probabilities": softmax_to_dict(probs.tolist(), labels),
                "model":         "bilstm",
            })
        return results
