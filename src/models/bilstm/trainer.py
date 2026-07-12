"""
src/models/bilstm/trainer.py
==============================
BiLSTM training loop with callbacks (EarlyStopping, ModelCheckpoint,
ReduceLROnPlateau) and training history persistence.
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path

from config.settings import (
    BILSTM_MODEL_PATH,
    BILSTM_HISTORY_PATH,
    BILSTM_BATCH_SIZE,
    BILSTM_EPOCHS,
    BILSTM_PATIENCE,
    BILSTM_LEARNING_RATE,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from src.preprocessing.pipeline import load_split
from src.preprocessing.tokenizer import BiLSTMTokenizer
from src.models.bilstm.architecture import build_bilstm_model
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def get_callbacks(model_path: Path = BILSTM_MODEL_PATH) -> list:
    """
    Build Keras training callbacks.

    Returns:
        List of [EarlyStopping, ModelCheckpoint, ReduceLROnPlateau].
    """
    callbacks = [
        # Stop training when val_loss stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=BILSTM_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Save the best model weights
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Reduce LR when learning plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=str(BILSTM_MODEL_PATH.parent / "logs"),
            histogram_freq=1,
        ),
    ]
    return callbacks


def train_bilstm(
    epochs:     int   = BILSTM_EPOCHS,
    batch_size: int   = BILSTM_BATCH_SIZE,
    lr:         float = BILSTM_LEARNING_RATE,
) -> dict:
    """
    Full BiLSTM training pipeline.

    Loads processed train/val splits, builds model, trains with callbacks,
    saves the model and training history.

    Args:
        epochs:     Maximum training epochs.
        batch_size: Training batch size.
        lr:         Initial learning rate.

    Returns:
        Training history dict {metric: [values per epoch]}.
    """
    logger.info("=" * 60)
    logger.info("  BiLSTM TRAINING — START")
    logger.info("=" * 60)

    # ── Load Processed Data ───────────────────────────────────
    logger.info("Loading processed train/val splits...")
    train_data = load_split(TRAIN_DATA_PATH)
    val_data   = load_split(VAL_DATA_PATH)

    X_train, y_train = train_data["X_bilstm"], train_data["y_onehot"]
    X_val,   y_val   = val_data["X_bilstm"],   val_data["y_onehot"]

    logger.info(f"Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Val:   X={X_val.shape},   y={y_val.shape}")

    # ── Load Tokenizer to get vocab size ──────────────────────
    tokenizer = BiLSTMTokenizer.load()

    # ── Build Model ───────────────────────────────────────────
    model = build_bilstm_model(vocab_size=tokenizer.vocab_size)
    model.optimizer.learning_rate.assign(lr)

    # ── Compute Class Weights (handles imbalance) ─────────────
    y_int = train_data["y_int"]
    unique, counts = np.unique(y_int, return_counts=True)
    total = len(y_int)
    class_weights = {
        int(cls): total / (len(unique) * cnt)
        for cls, cnt in zip(unique, counts)
    }
    logger.info(f"Class weights: {class_weights}")

    # ── Train ─────────────────────────────────────────────────
    logger.info(f"Training for up to {epochs} epochs (batch={batch_size}, lr={lr})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(),
        class_weight=class_weights,
        verbose=1,
    )

    # ── Save History ──────────────────────────────────────────
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    BILSTM_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BILSTM_HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)
    logger.info(f"Training history saved to: {BILSTM_HISTORY_PATH}")

    best_val_acc = max(history_dict.get("val_accuracy", [0]))
    logger.info("=" * 60)
    logger.info(f"  BiLSTM TRAINING — COMPLETE ✅")
    logger.info(f"  Best Val Accuracy: {best_val_acc:.4f}")
    logger.info(f"  Model saved to: {BILSTM_MODEL_PATH}")
    logger.info("=" * 60)

    return history_dict
