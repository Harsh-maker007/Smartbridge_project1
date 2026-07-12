"""
src/detection/ensemble.py
===========================
Weighted ensemble combiner for BiLSTM and BERT model outputs.
Merges probability distributions using configurable weights.
"""

from typing import Dict

from config.settings import BILSTM_ENSEMBLE_WEIGHT, BERT_ENSEMBLE_WEIGHT
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def weighted_average(
    bilstm_probs: Dict[str, float],
    bert_probs:   Dict[str, float],
    bilstm_weight: float = BILSTM_ENSEMBLE_WEIGHT,
    bert_weight:   float = BERT_ENSEMBLE_WEIGHT,
) -> Dict[str, float]:
    """
    Compute weighted average of BiLSTM and BERT probability distributions.

    Weights are normalized internally so they don't need to sum to 1.0,
    but it's recommended to set them such that they do (e.g., 0.4 + 0.6).

    Args:
        bilstm_probs:  {emotion: probability} from BiLSTM model.
        bert_probs:    {emotion: probability} from BERT model.
        bilstm_weight: Weight for BiLSTM (default 0.40).
        bert_weight:   Weight for BERT (default 0.60).

    Returns:
        Merged {emotion: probability} dict (sums to 1.0).
    """
    # Normalize weights
    total_weight = bilstm_weight + bert_weight
    w_bilstm = bilstm_weight / total_weight
    w_bert   = bert_weight   / total_weight

    # Compute weighted average for each emotion
    all_emotions = set(bilstm_probs.keys()) | set(bert_probs.keys())
    ensemble = {}
    for emotion in all_emotions:
        p_bilstm = bilstm_probs.get(emotion, 0.0)
        p_bert   = bert_probs.get(emotion, 0.0)
        ensemble[emotion] = w_bilstm * p_bilstm + w_bert * p_bert

    # Renormalize (float arithmetic safety)
    total = sum(ensemble.values())
    if total > 0:
        ensemble = {k: v / total for k, v in ensemble.items()}

    return ensemble


def get_top_emotion(probabilities: Dict[str, float]) -> tuple:
    """
    Get the emotion with the highest probability.

    Args:
        probabilities: {emotion: probability} dict.

    Returns:
        Tuple of (emotion_str, confidence_float).
    """
    top = max(probabilities, key=probabilities.get)
    return top, probabilities[top]
