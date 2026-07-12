"""
src/models/bert/trainer.py
============================
BERT fine-tuning pipeline using HuggingFace Trainer API.
Handles training arguments, compute_metrics, saving, and logging.
"""

import json
import numpy as np
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate

from config.settings import (
    BERT_BASE_MODEL,
    BERT_MODEL_PATH,
    BERT_METRICS_PATH,
    BERT_EPOCHS,
    BERT_BATCH_SIZE_TRAIN,
    BERT_BATCH_SIZE_EVAL,
    BERT_LEARNING_RATE,
    BERT_WARMUP_STEPS,
    BERT_WEIGHT_DECAY,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    NUM_CLASSES,
    EMOTION_LABELS,
    RANDOM_SEED,
)
from src.preprocessing.pipeline import load_split
from src.models.bert.architecture import build_bert_model
from src.models.bert.dataset import build_datasets
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def compute_metrics(eval_pred) -> dict:
    """
    Callback for HuggingFace Trainer to compute accuracy and F1.

    Args:
        eval_pred: EvalPrediction namedtuple (predictions, label_ids).

    Returns:
        Dict of {"accuracy": float, "f1": float}.
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric       = evaluate.load("f1")

    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1  = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    return {"accuracy": acc["accuracy"], "f1_macro": f1["f1"]}


def train_bert(
    model_name: str   = BERT_BASE_MODEL,
    epochs:     int   = BERT_EPOCHS,
    batch_size: int   = BERT_BATCH_SIZE_TRAIN,
    lr:         float = BERT_LEARNING_RATE,
) -> dict:
    """
    Full BERT fine-tuning pipeline.

    Args:
        model_name: HuggingFace model identifier.
        epochs:     Number of training epochs.
        batch_size: Training batch size (reduce to 8 if OOM on GPU).
        lr:         Learning rate.

    Returns:
        Dict of final training metrics.
    """
    logger.info("=" * 60)
    logger.info("  BERT FINE-TUNING — START")
    logger.info("=" * 60)

    # ── GPU / CPU detection ───────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device.upper()}")
    if device == "cpu":
        logger.warning("No GPU detected. Training will be slow. Consider using Colab/Kaggle GPU.")

    # ── Load Processed Data ───────────────────────────────────
    logger.info("Loading processed splits...")
    train_data = load_split(TRAIN_DATA_PATH)
    val_data   = load_split(VAL_DATA_PATH)

    # ── Build PyTorch Datasets ────────────────────────────────
    logger.info("Building PyTorch datasets...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    from src.models.bert.dataset import EmotionDataset
    train_ds = EmotionDataset(train_data["X_bert_texts"], train_data["y_int"].tolist(), tokenizer)
    val_ds   = EmotionDataset(val_data["X_bert_texts"],   val_data["y_int"].tolist(),   tokenizer)

    # ── Build Model ───────────────────────────────────────────
    model = build_bert_model(model_name=model_name)

    # ── Training Arguments ────────────────────────────────────
    BERT_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(BERT_MODEL_PATH),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=BERT_BATCH_SIZE_EVAL,
        learning_rate=lr,
        warmup_steps=BERT_WARMUP_STEPS,
        weight_decay=BERT_WEIGHT_DECAY,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(BERT_MODEL_PATH / "logs"),
        logging_steps=50,
        seed=RANDOM_SEED,
        fp16=(device == "cuda"),          # Mixed precision on GPU
        dataloader_num_workers=0,         # Windows compatibility
        report_to="none",                 # Disable wandb/tensorboard
    )

    # ── Trainer ───────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ─────────────────────────────────────────────────
    logger.info(f"Starting training for {epochs} epoch(s)...")
    train_result = trainer.train()

    # ── Save Model + Tokenizer ────────────────────────────────
    logger.info(f"Saving model to: {BERT_MODEL_PATH}")
    trainer.save_model(str(BERT_MODEL_PATH))
    tokenizer.save_pretrained(str(BERT_MODEL_PATH))

    # ── Evaluate on Validation ────────────────────────────────
    eval_results = trainer.evaluate()

    # ── Save Metrics ──────────────────────────────────────────
    metrics = {
        "train_loss":     train_result.training_loss,
        "val_accuracy":   eval_results.get("eval_accuracy", 0),
        "val_f1_macro":   eval_results.get("eval_f1_macro", 0),
        "train_runtime":  train_result.metrics.get("train_runtime", 0),
        "epochs":         epochs,
    }
    BERT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BERT_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {BERT_METRICS_PATH}")

    logger.info("=" * 60)
    logger.info(f"  BERT FINE-TUNING — COMPLETE ✅")
    logger.info(f"  Val Accuracy: {metrics['val_accuracy']:.4f}")
    logger.info(f"  Val F1 Macro: {metrics['val_f1_macro']:.4f}")
    logger.info("=" * 60)

    return metrics
