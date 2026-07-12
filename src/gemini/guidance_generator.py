"""
src/gemini/guidance_generator.py
==================================
Generates personalized learning guidance by calling the Gemini API.
Entry point for Streamlit: takes a detection result dict and returns
formatted guidance text.
"""

from src.gemini.client import GeminiClient
from src.gemini.prompt_builder import build_guidance_prompt
from src.utils.helpers import get_logger, truncate_text

logger = get_logger(__name__)

# Shared client instance
_client: GeminiClient = None


def _get_client() -> GeminiClient:
    """Return shared GeminiClient singleton."""
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client


def generate_guidance(detection_result: dict) -> dict:
    """
    Generate personalized learning guidance from a detection result.

    Args:
        detection_result: Output dict from EmotionDetector.detect().

    Returns:
        Dict with:
            - guidance_text:    Full markdown guidance string
            - guidance_excerpt: Short excerpt for CSV logging (100 chars)
            - emotion:          Final detected emotion
            - success:          Whether Gemini call succeeded
    """
    emotion    = detection_result.get("final_emotion", "confused")
    confidence = detection_result.get("final_confidence", 0.5)
    text       = detection_result.get("input_text", "")
    is_mixed   = detection_result.get("is_mixed", False)
    mixed_lbl  = detection_result.get("mixed_label", "")
    bilstm_em  = detection_result.get("bilstm", {}).get("emotion", "")
    bert_em    = detection_result.get("bert", {}).get("emotion", "")

    logger.info(f"Generating Gemini guidance for emotion: {emotion}")

    # Build prompt
    prompt = build_guidance_prompt(
        student_text=text,
        final_emotion=emotion,
        confidence=confidence,
        is_mixed=is_mixed,
        mixed_label=mixed_lbl,
        bilstm_emotion=bilstm_em,
        bert_emotion=bert_em,
    )

    # Call Gemini
    try:
        client = _get_client()
        guidance_text = client.generate(prompt)
        success = True
    except Exception as e:
        logger.error(f"Gemini guidance generation failed: {e}")
        guidance_text = GeminiClient._fallback_response()
        success = False

    # Create a short excerpt for logging
    excerpt = truncate_text(
        guidance_text.replace("#", "").replace("*", "").replace("\n", " "),
        max_chars=150,
    )

    return {
        "guidance_text":    guidance_text,
        "guidance_excerpt": excerpt,
        "emotion":          emotion,
        "success":          success,
    }
