"""
src/models/bilstm/architecture.py
====================================
BiLSTM model architecture definition using TensorFlow/Keras.

Architecture:
    Input → Embedding → SpatialDropout1D
    → Bidirectional(LSTM, return_sequences=True)
    → Bidirectional(LSTM)
    → Dropout
    → Dense(ReLU)
    → Dense(Softmax) → 5 emotion classes
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from config.settings import (
    BILSTM_MAX_VOCAB_SIZE,
    BILSTM_MAX_SEQUENCE_LEN,
    BILSTM_EMBEDDING_DIM,
    BILSTM_LSTM_UNITS_1,
    BILSTM_LSTM_UNITS_2,
    BILSTM_DENSE_UNITS,
    BILSTM_DROPOUT_RATE,
    BILSTM_SPATIAL_DROPOUT,
    NUM_CLASSES,
)
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def build_bilstm_model(
    vocab_size:      int   = None,
    max_length:      int   = BILSTM_MAX_SEQUENCE_LEN,
    embedding_dim:   int   = BILSTM_EMBEDDING_DIM,
    lstm_units_1:    int   = BILSTM_LSTM_UNITS_1,
    lstm_units_2:    int   = BILSTM_LSTM_UNITS_2,
    dense_units:     int   = BILSTM_DENSE_UNITS,
    dropout_rate:    float = BILSTM_DROPOUT_RATE,
    spatial_dropout: float = BILSTM_SPATIAL_DROPOUT,
    num_classes:     int   = NUM_CLASSES,
) -> tf.keras.Model:
    """
    Build and compile the BiLSTM emotion classifier.

    Args:
        vocab_size:      Vocabulary size (set from fitted tokenizer).
        max_length:      Input sequence length.
        embedding_dim:   Word embedding dimensions.
        lstm_units_1:    Units in the first BiLSTM layer.
        lstm_units_2:    Units in the second BiLSTM layer.
        dense_units:     Units in the intermediate Dense layer.
        dropout_rate:    Dropout rate for regularization.
        spatial_dropout: SpatialDropout1D rate after embedding.
        num_classes:     Number of output emotion classes.

    Returns:
        Compiled tf.keras.Model ready for training.
    """
    if vocab_size is None:
        vocab_size = BILSTM_MAX_VOCAB_SIZE

    model = models.Sequential([
        # ── Embedding Layer ───────────────────────────────────
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            name="embedding",
        ),

        # ── Spatial Dropout (drops entire feature maps) ───────
        layers.SpatialDropout1D(spatial_dropout, name="spatial_dropout"),

        # ── First BiLSTM Layer (returns full sequence) ────────
        layers.Bidirectional(
            layers.LSTM(
                units=lstm_units_1,
                return_sequences=True,
                recurrent_dropout=0.1,
                name="lstm_1",
            ),
            name="bilstm_1",
        ),

        # ── Second BiLSTM Layer (returns final state) ─────────
        layers.Bidirectional(
            layers.LSTM(
                units=lstm_units_2,
                return_sequences=False,
                recurrent_dropout=0.1,
                name="lstm_2",
            ),
            name="bilstm_2",
        ),

        # ── Regularization ────────────────────────────────────
        layers.Dropout(dropout_rate, name="dropout"),

        # ── Dense Classification Head ─────────────────────────
        layers.Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
            name="dense",
        ),
        layers.BatchNormalization(name="batch_norm"),
        layers.Dropout(dropout_rate / 2, name="dropout_2"),

        # ── Output Layer ──────────────────────────────────────
        layers.Dense(
            num_classes,
            activation="softmax",
            name="output",
        ),
    ], name="bilstm_emotion_classifier")

    # ── Compile ───────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("BiLSTM model built successfully.")
    model.summary(print_fn=logger.info)

    return model
