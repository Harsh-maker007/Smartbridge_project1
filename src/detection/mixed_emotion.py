"""
src/detection/mixed_emotion.py
================================
Mixed emotion detection — identifies when a student exhibits
multiple emotions simultaneously based on probability proximity.
"""

from typing import Dict, List, Tuple, Optional

from config.settings import MIXED_EMOTION_THRESHOLD
from src.utils.helpers import get_logger, top_k_emotions

logger = get_logger(__name__)


def detect_mixed_emotions(
    probabilities: Dict[str, float],
    threshold: float = MIXED_EMOTION_THRESHOLD,
    top_k: int = 2,
) -> Dict:
    """
    Detect whether the student is exhibiting mixed emotions.

    Mixed state is defined as: the probability gap between the top-1
    and top-2 emotions is less than the threshold.

    Example:
        frustrated: 0.45, confused: 0.38 → gap = 0.07 < 0.15 → MIXED ✓
        frustrated: 0.82, confused: 0.10 → gap = 0.72 > 0.15 → NOT MIXED ✗

    Args:
        probabilities: {emotion: probability} dict (ensemble output).
        threshold:     Max gap to still be considered "mixed".
        top_k:         How many top emotions to consider for mixing.

    Returns:
        Dict with:
            - is_mixed:        bool
            - mixed_emotions:  List of (emotion, prob) tuples if mixed, else []
            - primary_emotion: str — top emotion
            - gap:             float — probability gap between top-1 and top-2
    """
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    primary   = sorted_probs[0]
    secondary = sorted_probs[1] if len(sorted_probs) > 1 else None

    if secondary is None:
        return {
            "is_mixed":        False,
            "mixed_emotions":  [],
            "primary_emotion": primary[0],
            "gap":             1.0,
        }

    gap = primary[1] - secondary[1]
    is_mixed = gap < threshold

    mixed_emotions = []
    if is_mixed:
        mixed_emotions = [(e, p) for e, p in sorted_probs[:top_k]]
        logger.debug(
            f"Mixed emotions detected: {mixed_emotions} (gap={gap:.3f} < threshold={threshold})"
        )

    return {
        "is_mixed":        is_mixed,
        "mixed_emotions":  mixed_emotions,
        "primary_emotion": primary[0],
        "gap":             float(gap),
    }


def format_mixed_label(mixed_emotions: List[Tuple[str, float]]) -> str:
    """
    Format mixed emotions as a human-readable string.

    Args:
        mixed_emotions: List of (emotion, probability) tuples.

    Returns:
        String like "Frustrated + Confused" or empty string.
    """
    if not mixed_emotions:
        return ""
    labels = [e.capitalize() for e, _ in mixed_emotions]
    return " + ".join(labels)
