"""
src/models/bert/dataset.py
============================
PyTorch Dataset class for BERT fine-tuning.
Wraps the preprocessed BERT-cleaned texts + integer labels
into a format compatible with HuggingFace Trainer.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Optional

from config.settings import BERT_BASE_MODEL, BERT_MAX_LENGTH
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class EmotionDataset(Dataset):
    """
    PyTorch Dataset for emotion classification with BERT.

    Args:
        texts:      List of cleaned text strings.
        labels:     List of integer class labels (or None for inference).
        tokenizer:  HuggingFace AutoTokenizer instance.
        max_length: Maximum token sequence length.
    """

    def __init__(
        self,
        texts:      List[str],
        labels:     Optional[List[int]],
        tokenizer:  AutoTokenizer,
        max_length: int = BERT_MAX_LENGTH,
    ):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self._has_labels = labels is not None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_length,)
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self._has_labels:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def build_datasets(
    train_data: dict,
    val_data:   dict,
    test_data:  dict,
    model_name: str = BERT_BASE_MODEL,
    max_length: int = BERT_MAX_LENGTH,
) -> tuple:
    """
    Build PyTorch Datasets for train, val, test splits.

    Args:
        train_data: Processed train split dict (from load_split()).
        val_data:   Processed val split dict.
        test_data:  Processed test split dict.
        model_name: HuggingFace model name for tokenizer.
        max_length: Maximum token length.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    logger.info(f"Loading HuggingFace tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = EmotionDataset(
        texts=train_data["X_bert_texts"],
        labels=train_data["y_int"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_ds = EmotionDataset(
        texts=val_data["X_bert_texts"],
        labels=val_data["y_int"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_ds = EmotionDataset(
        texts=test_data["X_bert_texts"],
        labels=test_data["y_int"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    logger.info(f"Datasets built: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_ds, val_ds, test_ds
