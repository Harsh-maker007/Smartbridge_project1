"""
src/preprocessing/encoder.py
=============================
Label encoding utilities — converts emotion string labels to
integer indices and back. Wraps sklearn's LabelEncoder and
adds one-hot encoding support.
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from config.settings import LABEL_ENCODER_PATH, EMOTION_LABELS
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class EmotionLabelEncoder:
    """
    Wrapper around sklearn LabelEncoder for our 5 academic emotion classes.

    Provides:
        - fit / transform / inverse_transform
        - One-hot encoding (for BiLSTM)
        - Integer encoding (for BERT / CrossEntropyLoss)
        - Save / load to disk
    """

    def __init__(self, classes: list = None):
        self.encoder = LabelEncoder()
        self._fitted = False
        if classes:
            self.fit(classes)

    def fit(self, labels: list) -> "EmotionLabelEncoder":
        """
        Fit the encoder on a list of emotion string labels.

        Args:
            labels: List of emotion strings (e.g., ["bored", "confident", ...]).

        Returns:
            self (for chaining)
        """
        self.encoder.fit(labels)
        self._fitted = True
        logger.info(f"Encoder fitted. Classes: {list(self.encoder.classes_)}")
        return self

    def transform(self, labels: list) -> np.ndarray:
        """Convert string labels → integer indices."""
        self._check_fitted()
        return self.encoder.transform(labels)

    def inverse_transform(self, indices) -> list:
        """Convert integer indices → string labels."""
        self._check_fitted()
        return list(self.encoder.inverse_transform(indices))

    def to_onehot(self, labels: list, num_classes: int = None) -> np.ndarray:
        """
        Convert string labels → one-hot encoded array.

        Args:
            labels:      List of emotion strings.
            num_classes: Number of classes (defaults to len(encoder.classes_)).

        Returns:
            np.ndarray of shape (n_samples, num_classes).
        """
        self._check_fitted()
        indices = self.transform(labels)
        n = num_classes or len(self.encoder.classes_)
        return to_categorical(indices, num_classes=n)

    def index_to_label(self, idx: int) -> str:
        """Get emotion string from integer index."""
        self._check_fitted()
        return self.encoder.classes_[idx]

    def label_to_index(self, label: str) -> int:
        """Get integer index from emotion string."""
        self._check_fitted()
        return int(self.encoder.transform([label])[0])

    @property
    def classes(self) -> list:
        """Return list of class strings in order."""
        self._check_fitted()
        return list(self.encoder.classes_)

    @property
    def num_classes(self) -> int:
        """Return number of emotion classes."""
        self._check_fitted()
        return len(self.encoder.classes_)

    def save(self, path: Path = LABEL_ENCODER_PATH) -> None:
        """Persist the encoder to disk."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"LabelEncoder saved to: {path}")

    @classmethod
    def load(cls, path: Path = LABEL_ENCODER_PATH) -> "EmotionLabelEncoder":
        """Load a previously saved encoder from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LabelEncoder not found at: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"LabelEncoder loaded from: {path}")
        return obj

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call .fit() first.")


def build_default_encoder() -> EmotionLabelEncoder:
    """
    Build and return an encoder pre-fitted on the project's 5 emotion labels.
    Uses the canonical sorted order from config/settings.py.
    """
    return EmotionLabelEncoder(classes=sorted(EMOTION_LABELS))
