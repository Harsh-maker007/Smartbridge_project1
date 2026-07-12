"""
src/utils/validators.py
=======================
Input validation utilities for the emotion detection pipeline.
"""

from typing import Optional


def validate_text_input(text: str) -> tuple[bool, Optional[str]]:
    """
    Validate student text input before processing.

    Args:
        text: Raw text from the student.

    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str]).
        If valid, error_message is None.
    """
    if not text or not text.strip():
        return False, "Input text cannot be empty."

    word_count = len(text.strip().split())
    if word_count < 3:
        return False, f"Please enter at least 3 words (got {word_count})."

    if len(text) > 5000:
        return False, "Input text is too long. Please keep it under 5000 characters."

    return True, None


def validate_emotion_label(label: str, valid_labels: list) -> bool:
    """
    Check that a predicted emotion label is one of the valid classes.

    Args:
        label:        Predicted emotion string.
        valid_labels: List of valid emotion strings.

    Returns:
        True if label is valid.
    """
    return label.lower() in [l.lower() for l in valid_labels]


def validate_probabilities(probs: list, num_classes: int = 5) -> tuple[bool, Optional[str]]:
    """
    Validate that model output probabilities are well-formed.

    Args:
        probs:       List of float probabilities.
        num_classes: Expected number of classes.

    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str]).
    """
    if len(probs) != num_classes:
        return False, f"Expected {num_classes} probabilities, got {len(probs)}."

    if not all(0.0 <= p <= 1.0 for p in probs):
        return False, "All probabilities must be between 0 and 1."

    prob_sum = sum(probs)
    if not (0.95 <= prob_sum <= 1.05):
        return False, f"Probabilities must sum to ~1.0 (got {prob_sum:.4f})."

    return True, None
