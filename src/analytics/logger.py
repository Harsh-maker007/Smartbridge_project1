"""
src/analytics/logger.py
=========================
CSV interaction logger — appends every detection session to
data/interactions/interaction_logs.csv for analytics and auditing.
"""

import csv
import uuid
from pathlib import Path
from datetime import datetime

from config.settings import INTERACTION_LOG_PATH, LOG_COLUMNS
from src.utils.helpers import get_logger, get_timestamp

logger = get_logger(__name__)


def _ensure_log_file() -> None:
    """Create the CSV log file with headers if it doesn't exist."""
    INTERACTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not INTERACTION_LOG_PATH.exists():
        with open(INTERACTION_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()
        logger.info(f"Created interaction log: {INTERACTION_LOG_PATH}")


def log_interaction(
    detection_result: dict,
    guidance_excerpt: str = "",
    session_id:       str = None,
) -> str:
    """
    Append one interaction to the CSV log.

    Args:
        detection_result: Output from EmotionDetector.detect().
        guidance_excerpt: Short text excerpt from Gemini guidance.
        session_id:       Optional session ID; auto-generated if None.

    Returns:
        The session_id used for this log entry.
    """
    _ensure_log_file()
    session_id = session_id or str(uuid.uuid4())[:8]

    # Flatten keyword matches for CSV
    keywords_matched = detection_result.get("keywords_matched", {})
    keywords_str = "; ".join(
        f"{em}: {', '.join(kws)}" for em, kws in keywords_matched.items()
    ) if keywords_matched else ""

    mixed_emotions = detection_result.get("mixed_emotions", [])
    mixed_str = "; ".join(f"{e}({p:.2f})" for e, p in mixed_emotions) if mixed_emotions else ""

    row = {
        "timestamp":             detection_result.get("timestamp", get_timestamp()),
        "session_id":            session_id,
        "student_input":         detection_result.get("input_text", "")[:500],  # truncate
        "bilstm_emotion":        detection_result.get("bilstm", {}).get("emotion", ""),
        "bilstm_confidence":     detection_result.get("bilstm", {}).get("confidence", 0),
        "bert_emotion":          detection_result.get("bert", {}).get("emotion", ""),
        "bert_confidence":       detection_result.get("bert", {}).get("confidence", 0),
        "final_emotion":         detection_result.get("final_emotion", ""),
        "final_confidence":      detection_result.get("final_confidence", 0),
        "is_mixed":              detection_result.get("is_mixed", False),
        "mixed_emotions":        mixed_str,
        "rule_keywords_matched": keywords_str,
        "guidance_excerpt":      guidance_excerpt[:200],  # truncate
    }

    with open(INTERACTION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow(row)

    logger.debug(f"Logged interaction {session_id}: {row['final_emotion']}")
    return session_id
