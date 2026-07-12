"""
src/models/bert/evaluator.py
==============================
BERT model evaluation on the test set with classification report,
confusion matrix, and metrics persistence.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from config.settings import TEST_DATA_PATH, BERT_MODEL_DIR, EMOTION_LABELS
from src.preprocessing.pipeline import load_split
from src.models.bert.inference import BERTInference
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def evaluate_bert(save_plots: bool = True) -> dict:
    """
    Evaluate the fine-tuned BERT model on the test set.

    Args:
        save_plots: Whether to save confusion matrix PNG.

    Returns:
        Dict of metrics: accuracy, f1_macro, precision, recall, per_class.
    """
    logger.info("Evaluating BERT on test set...")

    test_data = load_split(TEST_DATA_PATH)
    texts  = test_data["X_bert_texts"]
    y_true = test_data["y_int"]

    # ── Load BERT Inference ───────────────────────────────────
    inferencer = BERTInference()
    inferencer.load()

    # Batch predict
    logger.info(f"Running inference on {len(texts)} test samples...")
    results = inferencer.predict_batch(texts, batch_size=32)
    y_pred  = np.array([r["pred_index"] for r in results])

    # ── Metrics ───────────────────────────────────────────────
    acc       = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)

    logger.info(f"\n{'='*50}")
    logger.info(f"  BERT Test Results")
    logger.info(f"{'='*50}")
    logger.info(f"  Accuracy:          {acc:.4f}")
    logger.info(f"  F1 (macro):        {f1_macro:.4f}")
    logger.info(f"  Precision (macro): {precision:.4f}")
    logger.info(f"  Recall (macro):    {recall:.4f}")
    logger.info(
        f"\nClassification Report:\n"
        + classification_report(y_true, y_pred, target_names=sorted(EMOTION_LABELS), zero_division=0)
    )

    cm = confusion_matrix(y_true, y_pred)
    if save_plots:
        _save_confusion_matrix(cm, sorted(EMOTION_LABELS), "bert")

    per_class = {}
    for i, emotion in enumerate(sorted(EMOTION_LABELS)):
        mask = y_true == i
        if mask.sum() > 0:
            per_class[emotion] = {
                "precision": float(precision_score(y_true == i, y_pred == i, zero_division=0)),
                "recall":    float(recall_score(y_true == i, y_pred == i, zero_division=0)),
                "f1":        float(f1_score(y_true == i, y_pred == i, zero_division=0)),
                "support":   int(mask.sum()),
            }

    metrics = {
        "accuracy":  float(acc),
        "f1_macro":  float(f1_macro),
        "precision": float(precision),
        "recall":    float(recall),
        "per_class": per_class,
    }

    metrics_path = BERT_MODEL_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")

    return metrics


def _save_confusion_matrix(cm: np.ndarray, labels: list, prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"{prefix.upper()} Confusion Matrix", fontsize=14, pad=15)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    path = BERT_MODEL_DIR / f"{prefix}_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to: {path}")
