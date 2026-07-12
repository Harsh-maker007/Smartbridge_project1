"""
src/preprocessing/pipeline.py
==============================
End-to-end preprocessing orchestrator.

Steps:
    1. Load raw CSV (from data/raw/emotions.csv)
    2. Remap labels to 5 academic emotions
    3. Clean texts (BiLSTM path + BERT path stored separately)
    4. Encode labels
    5. Train/validation/test split (stratified)
    6. Fit and apply BiLSTM tokenizer (on train set only)
    7. Save: train.pkl, val.pkl, test.pkl, tokenizer.pkl, label_encoder.pkl

Usage:
    python -c "from src.preprocessing.pipeline import run_preprocessing; run_preprocessing()"
    # or via: python scripts/train_bilstm.py (calls this internally)
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from config.settings import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    TOKENIZER_PATH,
    LABEL_ENCODER_PATH,
    EMOTION_LABELS,
    DATASET_SPLIT_TRAIN,
    DATASET_SPLIT_VAL,
    DATASET_SPLIT_TEST,
    RANDOM_SEED,
    PROCESSED_DATA_DIR,
)
from src.preprocessing.cleaner import clean_for_bilstm, clean_for_bert
from src.preprocessing.tokenizer import BiLSTMTokenizer
from src.preprocessing.encoder import EmotionLabelEncoder, build_default_encoder
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw remapped CSV produced by scripts/download_data.py.

    Args:
        path: Path to emotions.csv.

    Returns:
        DataFrame with columns: text, academic_emotion, original_label, split
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}.\n"
            f"Run: python scripts/download_data.py"
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to produce both BiLSTM and BERT text columns.

    Args:
        df: Raw DataFrame with 'text' and 'academic_emotion' columns.

    Returns:
        DataFrame with added 'text_bilstm' and 'text_bert' columns.
    """
    logger.info("Cleaning texts for BiLSTM (aggressive cleaning)...")
    df["text_bilstm"] = df["text"].apply(clean_for_bilstm)

    logger.info("Cleaning texts for BERT (minimal cleaning)...")
    df["text_bert"] = df["text"].apply(clean_for_bert)

    # Drop rows where cleaning produced empty strings
    before = len(df)
    df = df[df["text_bilstm"].str.strip().str.len() > 0].reset_index(drop=True)
    after = len(df)
    if before != after:
        logger.warning(f"Dropped {before - after} empty samples after cleaning.")

    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = DATASET_SPLIT_TRAIN,
    val_ratio:   float = DATASET_SPLIT_VAL,
    test_ratio:  float = DATASET_SPLIT_TEST,
    seed:        int   = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / validation / test split.

    Args:
        df:          Cleaned DataFrame.
        train_ratio: Fraction for training set.
        val_ratio:   Fraction for validation set.
        test_ratio:  Fraction for test set (must sum to 1.0).
        seed:        Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train/val/test ratios must sum to 1.0"

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df["academic_emotion"],
        random_state=seed,
    )

    # Second split: val vs test (from temp)
    val_fraction_of_temp = val_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_fraction_of_temp),
        stratify=temp_df["academic_emotion"],
        random_state=seed,
    )

    logger.info(
        f"Split complete → Train: {len(train_df):,} | "
        f"Val: {len(val_df):,} | Test: {len(test_df):,}"
    )

    # Log class distribution per split
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split["academic_emotion"].value_counts().to_dict()
        logger.info(f"  {name}: {dist}")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_and_save_tokenizer(
    train_texts: list,
    path: Path = TOKENIZER_PATH,
) -> BiLSTMTokenizer:
    """
    Fit BiLSTM tokenizer on training texts and save to disk.

    IMPORTANT: Tokenizer is fitted ONLY on training data to prevent data leakage.

    Args:
        train_texts: List of cleaned training texts (BiLSTM path).
        path:        Save path for tokenizer.pkl.

    Returns:
        Fitted BiLSTMTokenizer instance.
    """
    tokenizer = BiLSTMTokenizer()
    tokenizer.fit(train_texts)
    tokenizer.save(path)
    return tokenizer


def tokenize_and_encode(
    df: pd.DataFrame,
    tokenizer: BiLSTMTokenizer,
    encoder: EmotionLabelEncoder,
) -> dict:
    """
    Apply tokenization and label encoding to a DataFrame split.

    Args:
        df:        DataFrame with 'text_bilstm', 'text_bert', 'academic_emotion'.
        tokenizer: Fitted BiLSTMTokenizer.
        encoder:   Fitted EmotionLabelEncoder.

    Returns:
        Dict with keys:
            - X_bilstm:     np.ndarray (padded sequences for BiLSTM)
            - X_bert_texts: list of str (cleaned texts for BERT tokenizer)
            - y_int:        np.ndarray (integer labels for BERT/CrossEntropy)
            - y_onehot:     np.ndarray (one-hot labels for BiLSTM/Softmax)
            - texts_raw:    list of original text strings
            - labels:       list of emotion strings
    """
    X_bilstm     = tokenizer.transform(df["text_bilstm"].tolist())
    X_bert_texts = df["text_bert"].tolist()
    labels       = df["academic_emotion"].tolist()
    y_int        = encoder.transform(labels)
    y_onehot     = encoder.to_onehot(labels)

    return {
        "X_bilstm":     X_bilstm,
        "X_bert_texts": X_bert_texts,
        "y_int":        y_int,
        "y_onehot":     y_onehot,
        "texts_raw":    df["text"].tolist(),
        "labels":       labels,
    }


def save_split(data: dict, path: Path) -> None:
    """Save a processed split dict to a .pkl file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Saved processed split to: {path}")


def load_split(path: Path) -> dict:
    """Load a processed split dict from a .pkl file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found at: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Loaded split from: {path}")
    return data


def run_preprocessing(force: bool = False) -> None:
    """
    Full preprocessing pipeline entry point.

    Args:
        force: If True, re-run even if processed files already exist.
    """
    # Skip if already done (unless forced)
    if not force and all(p.exists() for p in [TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH]):
        logger.info("✅ Processed data already exists. Skipping preprocessing. (Use force=True to redo)")
        return

    logger.info("=" * 60)
    logger.info("  PREPROCESSING PIPELINE — START")
    logger.info("=" * 60)

    # Step 1: Load raw data
    df = load_raw_data()

    # Step 2: Clean texts
    df = clean_dataset(df)

    # Step 3: Split
    train_df, val_df, test_df = split_data(df)

    # Step 4: Fit label encoder (on ALL data classes for completeness)
    logger.info("Fitting label encoder...")
    encoder = build_default_encoder()
    encoder.save()

    # Step 5: Fit BiLSTM tokenizer on TRAIN ONLY
    logger.info("Fitting BiLSTM tokenizer on training data...")
    tokenizer = build_and_save_tokenizer(train_df["text_bilstm"].tolist())

    # Step 6: Tokenize and encode all splits
    logger.info("Tokenizing and encoding all splits...")
    train_data = tokenize_and_encode(train_df, tokenizer, encoder)
    val_data   = tokenize_and_encode(val_df,   tokenizer, encoder)
    test_data  = tokenize_and_encode(test_df,  tokenizer, encoder)

    # Step 7: Save to disk
    save_split(train_data, TRAIN_DATA_PATH)
    save_split(val_data,   VAL_DATA_PATH)
    save_split(test_data,  TEST_DATA_PATH)

    logger.info("=" * 60)
    logger.info("  PREPROCESSING PIPELINE — COMPLETE ✅")
    logger.info(f"  Train:    {len(train_data['labels']):,} samples")
    logger.info(f"  Val:      {len(val_data['labels']):,} samples")
    logger.info(f"  Test:     {len(test_data['labels']):,} samples")
    logger.info(f"  Vocab:    {tokenizer.vocab_size:,} tokens")
    logger.info(f"  Classes:  {encoder.classes}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_preprocessing()
