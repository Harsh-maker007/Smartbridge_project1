"""
scripts/download_data.py
========================
Downloads the dair-ai/emotion dataset from HuggingFace,
remaps emotion labels to our 5 academic emotions,
and saves the raw combined CSV to data/raw/emotions.csv.

Usage:
    python scripts/download_data.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from datasets import load_dataset
from config.settings import RAW_DATA_PATH, EMOTION_REMAP
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def download_and_remap():
    """
    Download dair-ai/emotion dataset from HuggingFace and remap labels.

    The dair-ai/emotion dataset has 6 classes:
        0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise

    We remap these to 5 academic learning emotions as defined in EMOTION_REMAP.
    """
    logger.info("📥 Downloading dair-ai/emotion dataset from HuggingFace...")
    dataset = load_dataset("dair-ai/emotion", trust_remote_code=True)

    # Integer → string label map from HuggingFace
    int_to_str = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise",
    }

    all_dfs = []
    for split_name in ["train", "validation", "test"]:
        if split_name not in dataset:
            logger.warning(f"Split '{split_name}' not found in dataset, skipping.")
            continue

        split = dataset[split_name]
        df = pd.DataFrame({
            "text":  split["text"],
            "label": [int_to_str[l] for l in split["label"]],
        })
        df["split"] = split_name
        all_dfs.append(df)
        logger.info(f"  {split_name}: {len(df)} samples")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Remap general emotions → academic emotions
    logger.info("🔀 Remapping emotion labels to academic categories...")
    combined["academic_emotion"] = combined["label"].map(EMOTION_REMAP)

    # Rename for clarity
    combined = combined.rename(columns={"label": "original_label"})

    # Show class distribution
    logger.info("\n📊 Academic Emotion Distribution:")
    dist = combined["academic_emotion"].value_counts()
    for emotion, count in dist.items():
        logger.info(f"  {emotion:>12}: {count:>6} samples")

    # Save
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"\n✅ Dataset saved to: {RAW_DATA_PATH}")
    logger.info(f"   Total samples: {len(combined)}")

    return combined


if __name__ == "__main__":
    download_and_remap()
