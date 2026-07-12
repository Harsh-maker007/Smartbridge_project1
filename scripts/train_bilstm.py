"""
scripts/train_bilstm.py
=======================
CLI entry point to train the BiLSTM emotion classifier.
Run after preprocessing: python scripts/download_data.py

Usage:
    python scripts/train_bilstm.py [--epochs N] [--batch-size N]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BiLSTM Emotion Classifier")
    parser.add_argument("--epochs",     type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr",         type=float, default=None, help="Learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("🚀 Starting BiLSTM training pipeline...")

    # Override settings if CLI args provided
    overrides = {}
    if args.epochs:     overrides["epochs"] = args.epochs
    if args.batch_size: overrides["batch_size"] = args.batch_size
    if args.lr:         overrides["lr"] = args.lr

    # Import here to allow settings override before import
    from src.preprocessing.pipeline import run_preprocessing
    from src.models.bilstm.trainer import train_bilstm

    logger.info("Step 1/3: Running preprocessing...")
    run_preprocessing()

    logger.info("Step 2/3: Training BiLSTM model...")
    history = train_bilstm(**overrides)

    logger.info("Step 3/3: Training complete!")
    logger.info("📁 Model saved to: models/bilstm/bilstm_model.h5")


if __name__ == "__main__":
    main()
