"""
src/preprocessing/cleaner.py
=============================
Text cleaning utilities for the emotion detection pipeline.
Handles HTML removal, lowercasing, contraction expansion,
punctuation stripping, and stopword filtering.
"""

import re
import string
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.utils.helpers import get_logger

logger = get_logger(__name__)

# ── Download required NLTK data ───────────────────────────────
def _ensure_nltk():
    """Download NLTK resources if not already present."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt",  "punkt"),
        ("corpora/wordnet",   "wordnet"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)

_ensure_nltk()

# ── Module-level singletons ───────────────────────────────────
_STOP_WORDS  = set(stopwords.words("english"))
_LEMMATIZER  = WordNetLemmatizer()

# Negation words to preserve (do NOT remove these stop words)
_NEGATION_WORDS = {
    "no", "not", "never", "neither", "nor", "nothing",
    "nobody", "nowhere", "cannot", "can't", "won't", "don't",
    "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
    "hadn't", "couldn't", "shouldn't", "wouldn't",
}
_FILTERED_STOP_WORDS = _STOP_WORDS - _NEGATION_WORDS


def remove_html(text: str) -> str:
    """Strip HTML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def expand_contractions(text: str) -> str:
    """Expand contractions (e.g., 'can't' → 'cannot')."""
    try:
        return contractions.fix(text)
    except Exception:
        return text


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)


def remove_mentions_hashtags(text: str) -> str:
    """Remove @mentions and #hashtags."""
    return re.sub(r"@\w+|#\w+", " ", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation but preserve apostrophes in contractions."""
    # Keep letters, digits, spaces
    return re.sub(r"[^\w\s]", " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces into one and strip."""
    return re.sub(r"\s+", " ", text).strip()


def remove_numbers(text: str) -> str:
    """Remove standalone numbers but keep alphanumeric tokens."""
    return re.sub(r"\b\d+\b", " ", text)


def remove_stopwords(text: str) -> str:
    """Remove stopwords (preserving negations)."""
    tokens = text.split()
    filtered = [t for t in tokens if t not in _FILTERED_STOP_WORDS]
    return " ".join(filtered)


def lemmatize(text: str) -> str:
    """Lemmatize each token in the text."""
    tokens = text.split()
    lemmatized = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(lemmatized)


def clean_text(
    text: str,
    remove_stops: bool = True,
    do_lemmatize: bool = True,
) -> str:
    """
    Full text cleaning pipeline:
      1. Remove HTML tags
      2. Expand contractions
      3. Remove URLs, mentions, hashtags
      4. Lowercase
      5. Remove punctuation
      6. Remove standalone numbers
      7. Normalize whitespace
      8. (Optional) Remove stopwords — preserving negations
      9. (Optional) Lemmatize

    Args:
        text:          Raw input string.
        remove_stops:  Whether to remove stopwords.
        do_lemmatize:  Whether to lemmatize tokens.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = remove_html(text)
    text = expand_contractions(text)
    text = remove_urls(text)
    text = remove_mentions_hashtags(text)
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = normalize_whitespace(text)

    if remove_stops:
        text = remove_stopwords(text)

    if do_lemmatize:
        text = lemmatize(text)

    return normalize_whitespace(text)


def clean_for_bilstm(text: str) -> str:
    """
    Clean text for BiLSTM — uses full cleaning with stopword removal + lemmatization.
    Produces shorter, stemmed sequences for the word-embedding model.
    """
    return clean_text(text, remove_stops=True, do_lemmatize=True)


def clean_for_bert(text: str) -> str:
    """
    Clean text for BERT — minimal cleaning only.
    BERT's tokenizer handles subword encoding; over-cleaning hurts performance.
    Keeps contractions, punctuation context, and full sentences.
    """
    text = remove_html(text)
    text = expand_contractions(text)
    text = remove_urls(text)
    text = remove_mentions_hashtags(text)
    text = normalize_whitespace(text)
    # BERT is case-insensitive (bert-base-uncased handles it internally)
    return text
