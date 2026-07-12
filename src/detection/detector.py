"""
src/detection/detector.py
===========================
Main inference pipeline orchestrator.
Coordinates: BiLSTM → BERT → Rule-Based Boost → Ensemble → Mixed Detection.
Returns a structured result dict for the Streamlit app and analytics logger.
"""

from typing import Optional

from src.models.bilstm.inference import BiLSTMInference
from src.models.bert.inference import BERTInference
from src.detection.rule_based import run_rule_based
from src.detection.ensemble import weighted_average, get_top_emotion
from src.detection.mixed_emotion import detect_mixed_emotions, format_mixed_label
from config.emotions import EMOTION_META
from src.utils.helpers import get_logger, get_timestamp

logger = get_logger(__name__)


class EmotionDetector:
    """
    Unified emotion detection pipeline.

    Loads both models once (lazy initialization), then provides
    a single .detect(text) method for the Streamlit app.

    Usage:
        detector = EmotionDetector()
        result = detector.detect("I've been stuck on this problem for hours")
    """

    def __init__(self):
        self._bilstm: Optional[BiLSTMInference] = None
        self._bert:   Optional[BERTInference]   = None
        self._loaded  = False

    def load(self) -> "EmotionDetector":
        """Lazy-load both models."""
        if self._loaded:
            return self

        logger.info("Loading BiLSTM model...")
        self._bilstm = BiLSTMInference().load()

        logger.info("Loading BERT model...")
        self._bert = BERTInference().load()

        self._loaded = True
        logger.info("✅ Both models loaded and ready.")
        return self

    def detect(self, text: str) -> dict:
        """
        Run the full emotion detection pipeline on student text.

        Pipeline:
            1. BiLSTM prediction  → probabilities
            2. BERT prediction    → probabilities
            3. Weighted ensemble  → merged probabilities
            4. Rule-based boost   → keyword-adjusted probabilities
            5. Mixed detection    → flag if ambiguous
            6. Final emotion      → top class + confidence

        Args:
            text: Raw student input text.

        Returns:
            Structured result dict with all intermediate outputs.
        """
        if not self._loaded:
            self.load()

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        logger.info(f"Detecting emotion for: '{text[:60]}...'")

        # ── Step 1: Model Predictions ─────────────────────────
        bilstm_result = self._bilstm.predict(text)
        bert_result   = self._bert.predict(text)

        # ── Step 2: Weighted Ensemble ─────────────────────────
        ensemble_probs = weighted_average(
            bilstm_probs=bilstm_result["probabilities"],
            bert_probs=bert_result["probabilities"],
        )

        # ── Step 3: Rule-Based Keyword Enhancement ────────────
        rule_output = run_rule_based(text, ensemble_probs)
        final_probs = rule_output["probabilities"]

        # ── Step 4: Mixed Emotion Detection ───────────────────
        mixed_output = detect_mixed_emotions(final_probs)

        # ── Step 5: Final Decision ────────────────────────────
        final_emotion, final_confidence = get_top_emotion(final_probs)
        emotion_meta = EMOTION_META.get(final_emotion, {})

        result = {
            # ── Final Output ──────────────────────────────────
            "final_emotion":    final_emotion,
            "final_confidence": round(final_confidence, 4),
            "final_label":      emotion_meta.get("label", final_emotion.capitalize()),
            "emoji":            emotion_meta.get("emoji", "🔵"),
            "color_hex":        emotion_meta.get("color_hex", "#6B7280"),

            # ── Per-Model Results ─────────────────────────────
            "bilstm": {
                "emotion":       bilstm_result["emotion"],
                "confidence":    round(bilstm_result["confidence"], 4),
                "probabilities": bilstm_result["probabilities"],
            },
            "bert": {
                "emotion":       bert_result["emotion"],
                "confidence":    round(bert_result["confidence"], 4),
                "probabilities": bert_result["probabilities"],
            },

            # ── Ensemble Probabilities ────────────────────────
            "ensemble_probabilities": {k: round(v, 4) for k, v in final_probs.items()},

            # ── Rule-Based Output ─────────────────────────────
            "rule_applied":     rule_output["rule_applied"],
            "keywords_matched": rule_output["keywords_matched"],

            # ── Mixed Emotion ─────────────────────────────────
            "is_mixed":        mixed_output["is_mixed"],
            "mixed_emotions":  mixed_output["mixed_emotions"],
            "mixed_label":     format_mixed_label(mixed_output["mixed_emotions"]),
            "emotion_gap":     round(mixed_output["gap"], 4),

            # ── Models Agree? ─────────────────────────────────
            "models_agree": bilstm_result["emotion"] == bert_result["emotion"],

            # ── Metadata ─────────────────────────────────────
            "input_text": text,
            "timestamp":  get_timestamp(),
        }

        logger.info(
            f"Detection complete: {final_emotion} ({final_confidence:.1%}) "
            f"| BiLSTM={bilstm_result['emotion']} BERT={bert_result['emotion']} "
            f"| Mixed={mixed_output['is_mixed']} | Rules={rule_output['rule_applied']}"
        )

        return result


# ── Singleton instance for reuse across Streamlit sessions ────
_detector_instance: Optional[EmotionDetector] = None

def get_detector() -> EmotionDetector:
    """Return shared EmotionDetector singleton (loads models once)."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = EmotionDetector()
        _detector_instance.load()
    return _detector_instance
