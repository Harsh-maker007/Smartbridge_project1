"""
src/models/bert/architecture.py
=================================
BERT model architecture using HuggingFace Transformers.
Loads bert-base-uncased and adds a classification head for 5 emotions.
"""

from transformers import AutoModelForSequenceClassification, AutoConfig
from config.settings import BERT_BASE_MODEL, NUM_CLASSES, EMOTION_LABELS
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def build_bert_model(
    model_name:  str = BERT_BASE_MODEL,
    num_labels:  int = NUM_CLASSES,
) -> AutoModelForSequenceClassification:
    """
    Build a BERT classification model for emotion detection.

    Loads bert-base-uncased from HuggingFace Hub and adds a linear
    classification head on top of the [CLS] token representation.

    Architecture:
        Input (input_ids, attention_mask)
            → BERT Encoder (12 layers, 768 hidden, 12 heads)
            → [CLS] pooled output (768,)
            → Dropout(0.1)
            → Linear(768 → 5)
            → Softmax

    Args:
        model_name: HuggingFace model identifier.
        num_labels: Number of output emotion classes.

    Returns:
        AutoModelForSequenceClassification (PyTorch nn.Module)
    """
    # Build label-to-id and id-to-label mappings for HuggingFace
    sorted_labels = sorted(EMOTION_LABELS)
    id2label = {i: label for i, label in enumerate(sorted_labels)}
    label2id = {label: i for i, label in id2label.items()}

    logger.info(f"Loading BERT model: {model_name} (num_labels={num_labels})")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,   # Safety for custom num_labels
    )

    # Log parameter count
    total_params  = sum(p.numel() for p in model.parameters())
    trainable     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters:    {total_params:,}")
    logger.info(f"Trainable parameters: {trainable:,}")

    return model
