"""
src/models/bert/inference.py
==============================
BERT inference engine — loads the saved fine-tuned model and
tokenizer, runs forward pass, and returns emotion probabilities.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, List

from config.settings import BERT_MODEL_PATH, BERT_MAX_LENGTH, EMOTION_LABELS
from src.preprocessing.cleaner import clean_for_bert
from src.utils.helpers import get_logger, softmax_to_dict

logger = get_logger(__name__)


class BERTInference:
    """
    Inference engine for the fine-tuned BERT emotion classifier.

    Usage:
        inferencer = BERTInference()
        inferencer.load()
        result = inferencer.predict("I have no idea what's happening")
        # {"emotion": "confused", "confidence": 0.91, "probabilities": {...}}
    """

    def __init__(self, model_path=None):
        self.model_path = model_path or BERT_MODEL_PATH
        self.model:     Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer]                      = None
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded    = False

    def load(self) -> "BERTInference":
        """Load BERT model and tokenizer from disk."""
        if self._loaded:
            return self

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"BERT model not found at {self.model_path}.\n"
                f"Run: python scripts/train_bert.py"
            )

        logger.info(f"Loading BERT model from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model     = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info(f"BERT model loaded on {str(self.device).upper()}.")
        return self

    def predict(self, text: str) -> dict:
        """
        Predict emotion for a single text.

        Args:
            text: Raw student input text.

        Returns:
            Dict with emotion, confidence, probabilities, pred_index.
        """
        if not self._loaded:
            self.load()

        cleaned = clean_for_bert(text)
        encoding = self.tokenizer(
            cleaned,
            max_length=BERT_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Move to device
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

        labels      = sorted(EMOTION_LABELS)
        top_idx     = int(np.argmax(probs))
        top_emotion = labels[top_idx]
        confidence  = float(probs[top_idx])

        return {
            "emotion":       top_emotion,
            "confidence":    confidence,
            "probabilities": softmax_to_dict(probs.tolist(), labels),
            "pred_index":    top_idx,
            "model":         "bert",
        }

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[dict]:
        """
        Predict emotions for a batch of texts efficiently.

        Args:
            texts:      List of raw text strings.
            batch_size: Number of texts per forward pass.

        Returns:
            List of prediction dicts.
        """
        if not self._loaded:
            self.load()

        labels  = sorted(EMOTION_LABELS)
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = [clean_for_bert(t) for t in texts[i:i + batch_size]]
            encoding = self.tokenizer(
                batch_texts,
                max_length=BERT_MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids      = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs   = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs_all = F.softmax(outputs.logits, dim=-1).cpu().numpy()

            for probs in probs_all:
                top_idx = int(np.argmax(probs))
                results.append({
                    "emotion":       labels[top_idx],
                    "confidence":    float(probs[top_idx]),
                    "probabilities": softmax_to_dict(probs.tolist(), labels),
                    "pred_index":    top_idx,
                    "model":         "bert",
                })

        return results
