"""
scripts/evaluate_models.py
==========================
CLI script to evaluate and compare BiLSTM and BERT models on the test set.
Prints a side-by-side comparison of accuracy, F1, precision, and recall.

Usage:
    python scripts/evaluate_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def main():
    logger.info("📊 Evaluating models on test set...")

    from src.models.bilstm.evaluator import evaluate_bilstm
    from src.models.bert.evaluator import evaluate_bert

    logger.info("\n── BiLSTM Evaluation ──")
    bilstm_metrics = evaluate_bilstm()

    logger.info("\n── BERT Evaluation ──")
    bert_metrics = evaluate_bert()

    logger.info("\n── Comparison Summary ──")
    headers = ["Metric", "BiLSTM", "BERT"]
    rows = [
        ["Accuracy",  f"{bilstm_metrics.get('accuracy', 0):.4f}",  f"{bert_metrics.get('accuracy', 0):.4f}"],
        ["F1 (macro)", f"{bilstm_metrics.get('f1_macro', 0):.4f}", f"{bert_metrics.get('f1_macro', 0):.4f}"],
        ["Precision",  f"{bilstm_metrics.get('precision', 0):.4f}", f"{bert_metrics.get('precision', 0):.4f}"],
        ["Recall",     f"{bilstm_metrics.get('recall', 0):.4f}",    f"{bert_metrics.get('recall', 0):.4f}"],
    ]

    # Print table
    col_w = [max(len(r[i]) for r in [headers] + rows) + 2 for i in range(3)]
    def fmt_row(r): return "| " + " | ".join(cell.ljust(col_w[i]) for i, cell in enumerate(r)) + " |"
    sep = "|-" + "-|-".join("-" * w for w in col_w) + "-|"
    logger.info(fmt_row(headers))
    logger.info(sep)
    for row in rows:
        logger.info(fmt_row(row))


if __name__ == "__main__":
    main()
