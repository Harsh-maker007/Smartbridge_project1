"""
scripts/train_bert.py
=====================
CLI entry point to fine-tune the BERT emotion classifier.
Run after preprocessing: python scripts/download_data.py

Usage:
    python scripts/train_bert.py [--epochs N] [--batch-size N]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT Emotion Classifier")
    parser.add_argument("--epochs",     type=int,   default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int,   default=None, help="Training batch size")
    parser.add_argument("--lr",         type=float, default=None, help="Learning rate")
    parser.add_argument("--model-name", type=str,   default=None, help="HuggingFace model name")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("🚀 Starting BERT fine-tuning pipeline...")

    overrides = {}
    if args.epochs:     overrides["epochs"] = args.epochs
    if args.batch_size: overrides["batch_size"] = args.batch_size
    if args.lr:         overrides["lr"] = args.lr
    if args.model_name: overrides["model_name"] = args.model_name

    from src.preprocessing.pipeline import run_preprocessing
    from src.models.bert.trainer import train_bert

    logger.info("Step 1/3: Running preprocessing...")
    run_preprocessing()

    logger.info("Step 2/3: Fine-tuning BERT model...")
    results = train_bert(**overrides)

    logger.info("Step 3/3: Training complete!")
    logger.info("📁 Model saved to: models/bert/model/")


if __name__ == "__main__":
    main()
