"""
src/utils/helpers.py
====================
Common utility functions shared across the project.
"""

import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Logging Setup ─────────────────────────────────────────────
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Create and configure a logger with console and optional file output.

    Args:
        name:  Logger name (typically __name__ of calling module).
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ── ID / Timestamp Helpers ────────────────────────────────────
def generate_session_id() -> str:
    """Generate a unique session ID (UUID4)."""
    return str(uuid.uuid4())


def get_timestamp() -> str:
    """Return current timestamp as ISO 8601 string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Text Helpers ──────────────────────────────────────────────
def truncate_text(text: str, max_chars: int = 200) -> str:
    """
    Truncate text to max_chars and append '...' if needed.

    Args:
        text:      Input string.
        max_chars: Maximum character length.

    Returns:
        Truncated string.
    """
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."


def is_valid_input(text: str, min_words: int = 3) -> bool:
    """
    Check if user input has enough content to process.

    Args:
        text:      Input string.
        min_words: Minimum number of words required.

    Returns:
        True if input is valid.
    """
    return bool(text and len(text.strip().split()) >= min_words)


# ── File Helpers ──────────────────────────────────────────────
def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_exists(path: Path) -> bool:
    """Check if a file exists and is not empty."""
    path = Path(path)
    return path.exists() and path.stat().st_size > 0


# ── Probability Helpers ───────────────────────────────────────
def softmax_to_dict(probabilities: list, labels: list) -> dict:
    """
    Convert a list of probabilities to a dict keyed by label.

    Args:
        probabilities: List of float probabilities (must sum to ~1.0).
        labels:        List of class labels in the same order.

    Returns:
        Dict mapping label → probability.
    """
    return {label: float(prob) for label, prob in zip(labels, probabilities)}


def top_k_emotions(prob_dict: dict, k: int = 2) -> list:
    """
    Return top-k emotions sorted by descending probability.

    Args:
        prob_dict: Dict of {emotion: probability}.
        k:         Number of top emotions to return.

    Returns:
        List of (emotion, probability) tuples.
    """
    return sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:k]
