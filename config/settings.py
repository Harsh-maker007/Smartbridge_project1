"""
config/settings.py
==================
Central configuration for the AI Emotion Detection Platform.
All tunable hyperparameters, paths, and API settings live here.
Import this module wherever configuration is needed.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────
load_dotenv()

# ── Base Paths ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERACTIONS_DIR = DATA_DIR / "interactions"
MODELS_DIR = BASE_DIR / "models"
BILSTM_MODEL_DIR = MODELS_DIR / "bilstm"
BERT_MODEL_DIR = MODELS_DIR / "bert"

# ── Data Paths ────────────────────────────────────────────────
RAW_DATA_PATH = RAW_DATA_DIR / "emotions.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.pkl"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.pkl"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.pkl"
TOKENIZER_PATH = PROCESSED_DATA_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = PROCESSED_DATA_DIR / "label_encoder.pkl"
INTERACTION_LOG_PATH = INTERACTIONS_DIR / "interaction_logs.csv"

# ── Model Save Paths ──────────────────────────────────────────
BILSTM_MODEL_PATH = BILSTM_MODEL_DIR / "bilstm_model.h5"
BILSTM_HISTORY_PATH = BILSTM_MODEL_DIR / "training_history.json"
BERT_MODEL_PATH = BERT_MODEL_DIR / "model"
BERT_METRICS_PATH = BERT_MODEL_DIR / "training_metrics.json"

# ── Dataset Configuration ─────────────────────────────────────
DATASET_NAME = "dair-ai/emotion"          # HuggingFace dataset ID
DATASET_SPLIT_TRAIN = 0.80
DATASET_SPLIT_VAL = 0.10
DATASET_SPLIT_TEST = 0.10
RANDOM_SEED = 42

# ── Emotion Label Remapping (dair-ai/emotion → Academic) ──────
# dair-ai labels: sadness(0), joy(1), love(2), anger(3), fear(4), surprise(5)
EMOTION_REMAP = {
    "sadness":  "bored",
    "joy":      "confident",
    "love":     "confident",     # merged into confident
    "anger":    "frustrated",
    "fear":     "confused",
    "surprise": "curious",
}

# ── Target Emotion Labels (sorted alphabetically) ─────────────
EMOTION_LABELS = ["bored", "confident", "confused", "curious", "frustrated"]
NUM_CLASSES = len(EMOTION_LABELS)

# ── BiLSTM Hyperparameters ────────────────────────────────────
BILSTM_MAX_SEQUENCE_LEN = 100
BILSTM_MAX_VOCAB_SIZE = 20000
BILSTM_EMBEDDING_DIM = 128
BILSTM_LSTM_UNITS_1 = 128
BILSTM_LSTM_UNITS_2 = 64
BILSTM_DENSE_UNITS = 64
BILSTM_DROPOUT_RATE = 0.3
BILSTM_SPATIAL_DROPOUT = 0.2
BILSTM_BATCH_SIZE = 64
BILSTM_EPOCHS = 20
BILSTM_PATIENCE = 5           # EarlyStopping patience
BILSTM_LEARNING_RATE = 0.001

# ── BERT Hyperparameters ──────────────────────────────────────
BERT_BASE_MODEL = "bert-base-uncased"
BERT_MAX_LENGTH = 128
BERT_BATCH_SIZE_TRAIN = 16
BERT_BATCH_SIZE_EVAL = 32
BERT_EPOCHS = 4
BERT_LEARNING_RATE = 2e-5
BERT_WARMUP_STEPS = 500
BERT_WEIGHT_DECAY = 0.01

# ── Ensemble / Detection Settings ─────────────────────────────
BILSTM_ENSEMBLE_WEIGHT = float(os.getenv("BILSTM_WEIGHT", 0.40))
BERT_ENSEMBLE_WEIGHT = float(os.getenv("BERT_WEIGHT", 0.60))
MIXED_EMOTION_THRESHOLD = float(os.getenv("MIXED_EMOTION_THRESHOLD", 0.15))
RULE_BOOST_FACTOR = 0.15      # Confidence boost when keyword matched

# ── Gemini API ────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_MAX_TOKENS = 1024
GEMINI_TEMPERATURE = 0.7

# ── Logging ───────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Interaction Log CSV Columns ───────────────────────────────
LOG_COLUMNS = [
    "timestamp",
    "session_id",
    "student_input",
    "bilstm_emotion",
    "bilstm_confidence",
    "bert_emotion",
    "bert_confidence",
    "final_emotion",
    "final_confidence",
    "is_mixed",
    "mixed_emotions",
    "rule_keywords_matched",
    "guidance_excerpt",
]

# ── Ensure directories exist ──────────────────────────────────
def create_directories():
    """Create all required project directories if they don't exist."""
    dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        INTERACTIONS_DIR,
        BILSTM_MODEL_DIR,
        BERT_MODEL_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

# Auto-create on import
create_directories()
