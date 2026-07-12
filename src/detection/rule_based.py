"""
src/detection/rule_based.py
=============================
Rule-based keyword enhancement for emotion detection.
Scans student text for emotion-specific keywords and
applies a confidence boost to the corresponding emotion.
"""

import re
from typing import Dict, List, Tuple

from config.emotions import EMOTION_KEYWORDS
from config.settings import RULE_BOOST_FACTOR
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def find_matched_keywords(text: str) -> Dict[str, List[str]]:
    """
    Find all emotion keywords present in the text.

    Args:
        text: Raw or lightly cleaned student input.

    Returns:
        Dict mapping emotion → list of matched keyword phrases.
    """
    text_lower = text.lower()
    matches: Dict[str, List[str]] = {}

    for emotion, keywords in EMOTION_KEYWORDS.items():
        found = []
        for keyword in keywords:
            # Use word-boundary-aware matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                found.append(keyword)
        if found:
            matches[emotion] = found

    return matches


def apply_keyword_boost(
    probabilities: Dict[str, float],
    matched_keywords: Dict[str, List[str]],
    boost_factor: float = RULE_BOOST_FACTOR,
) -> Tuple[Dict[str, float], bool]:
    """
    Boost probability scores for emotions where keywords were found.

    Rules:
        1. For each emotion with matched keywords, add boost_factor.
        2. Renormalize so probabilities sum to 1.0.
        3. If no keywords matched, return original probabilities unchanged.

    Args:
        probabilities:    Dict of {emotion: probability} from model.
        matched_keywords: Output of find_matched_keywords().
        boost_factor:     Amount to add per matched emotion.

    Returns:
        Tuple of (boosted_probabilities, rule_was_applied).
    """
    if not matched_keywords:
        return probabilities, False

    boosted = dict(probabilities)

    # Apply boost
    for emotion in matched_keywords:
        if emotion in boosted:
            boosted[emotion] = min(1.0, boosted[emotion] + boost_factor)

    # Renormalize
    total = sum(boosted.values())
    if total > 0:
        boosted = {k: v / total for k, v in boosted.items()}

    logger.debug(f"Rule boost applied. Matched: {matched_keywords}")
    return boosted, True


def run_rule_based(
    text: str,
    probabilities: Dict[str, float],
) -> Dict:
    """
    Full rule-based enhancement pipeline.

    Args:
        text:          Raw student input text.
        probabilities: Combined model probability dict.

    Returns:
        Dict with:
            - probabilities:      Updated probability dict
            - keywords_matched:   {emotion: [keywords]} that triggered
            - rule_applied:       bool — whether any rule fired
    """
    matched = find_matched_keywords(text)
    boosted_probs, applied = apply_keyword_boost(probabilities, matched)

    return {
        "probabilities":    boosted_probs,
        "keywords_matched": matched,
        "rule_applied":     applied,
    }
