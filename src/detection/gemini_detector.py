"""
src/detection/gemini_detector.py
====================================
Gemini-based fallback emotion detector.
Used in cloud/demo mode when BiLSTM and BERT model files
are not available. Uses the Gemini API to classify emotion
AND generate guidance in a single call — keeping API usage minimal.
"""

import json
import re
from typing import Dict

import google.generativeai as genai
from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from config.emotions import EMOTION_META, EMOTION_KEYWORDS
from src.detection.rule_based import run_rule_based
from src.detection.mixed_emotion import detect_mixed_emotions, format_mixed_label
from src.utils.helpers import get_logger, get_timestamp, softmax_to_dict

logger = get_logger(__name__)

EMOTION_LABELS = ["bored", "confident", "confused", "curious", "frustrated"]


def _build_classify_prompt(text: str) -> str:
    return f"""You are an expert at detecting student learning emotions from text.

Analyze the following student's description and classify it into EXACTLY ONE of these emotions:
bored, confident, confused, curious, frustrated

Return ONLY a JSON object with this exact structure (no markdown, no explanation):
{{
  "bored": 0.10,
  "confident": 0.05,
  "confused": 0.15,
  "curious": 0.10,
  "frustrated": 0.60
}}

The values must be probabilities that sum to 1.0.
Base the classification on the emotional tone, word choice, and context.

Student's text: "{text}"

JSON response:"""


class GeminiEmotionDetector:
    """
    Lightweight emotion detector using Gemini API.
    Used as a fallback when trained ML models are not available.
    """

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured in .env")
        genai.configure(api_key=GEMINI_API_KEY)
        self._model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,   # Low temperature for consistent classification
                max_output_tokens=200,
            ),
        )

    def detect(self, text: str) -> dict:
        """
        Classify emotion using Gemini, then apply rule-based boosting.

        Args:
            text: Student's raw input text.

        Returns:
            Detection result dict (same structure as EmotionDetector.detect()).
        """
        # Step 1: Gemini classification
        try:
            prompt   = _build_classify_prompt(text)
            response = self._model.generate_content(prompt)
            raw      = response.text.strip()

            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            if json_match:
                probs_raw = json.loads(json_match.group())
                # Normalize
                total  = sum(float(v) for v in probs_raw.values())
                probs  = {k: float(v) / total for k, v in probs_raw.items() if k in EMOTION_LABELS}
            else:
                probs = _fallback_probs()
        except Exception as e:
            logger.warning(f"Gemini classification failed: {e}. Using rule-based fallback.")
            probs = _fallback_probs()

        # Step 2: Rule-based keyword boost
        rule_output    = run_rule_based(text, probs)
        final_probs    = rule_output["probabilities"]

        # Step 3: Mixed emotion detection
        mixed_output   = detect_mixed_emotions(final_probs)

        # Step 4: Final decision
        top_emotion    = max(final_probs, key=final_probs.get)
        top_confidence = final_probs[top_emotion]
        meta           = EMOTION_META.get(top_emotion, {})

        return {
            "final_emotion":           top_emotion,
            "final_confidence":        round(top_confidence, 4),
            "final_label":             meta.get("label", top_emotion.capitalize()),
            "emoji":                   meta.get("emoji", "🔵"),
            "color_hex":               meta.get("color_hex", "#6B7280"),
            "bilstm":                  {"emotion": top_emotion, "confidence": round(top_confidence, 4), "probabilities": final_probs},
            "bert":                    {"emotion": top_emotion, "confidence": round(top_confidence, 4), "probabilities": final_probs},
            "ensemble_probabilities":  {k: round(v, 4) for k, v in final_probs.items()},
            "rule_applied":            rule_output["rule_applied"],
            "keywords_matched":        rule_output["keywords_matched"],
            "is_mixed":                mixed_output["is_mixed"],
            "mixed_emotions":          mixed_output["mixed_emotions"],
            "mixed_label":             format_mixed_label(mixed_output["mixed_emotions"]),
            "emotion_gap":             round(mixed_output["gap"], 4),
            "models_agree":            True,
            "input_text":              text,
            "timestamp":               get_timestamp(),
            "demo_mode":               True,
        }


def _fallback_probs() -> Dict[str, float]:
    """Return uniform distribution as a safe fallback."""
    n = len(EMOTION_LABELS)
    return {e: 1.0 / n for e in EMOTION_LABELS}


# ── Singleton ─────────────────────────────────────────────────
_gemini_detector = None

def get_gemini_detector() -> GeminiEmotionDetector:
    global _gemini_detector
    if _gemini_detector is None:
        _gemini_detector = GeminiEmotionDetector()
    return _gemini_detector
