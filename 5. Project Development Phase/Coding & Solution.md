# Coding & Solution

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 3 Marks |

---

## Solution Implementation Summary

The Emotion Detector platform was built entirely in Python using a modular, layered architecture. Below is a summary of how each major feature was implemented.

---

## Module 1: Data Preprocessing (`src/preprocessing/`)

**Dataset:** `dair-ai/emotion` from HuggingFace Hub (16K training samples, 6 original classes remapped to 5)

| Original Class | Mapped To |
|---|---|
| joy | Confident |
| sadness | Frustrated |
| anger | Frustrated |
| fear | Confused |
| love | Curious |
| surprise | Curious |

**Preprocessing Pipeline:**
```
Raw Text → HTML Removal → URL/Mention Removal → Lowercase
         → Punctuation Removal → Stopword Removal (negation-preserving)
         → Lemmatization (WordNet) → Padded Integer Sequence
```

**BERT Path:** Minimal cleaning (preserve casing, punctuation context) → BertTokenizer subword encoding

---

## Module 2: BiLSTM Model (`src/models/bilstm/`)

```python
# Model Architecture (Keras)
Input: (100,)  # padded integer sequence
→ Embedding(10000, 128, input_length=100)
→ SpatialDropout1D(0.3)
→ Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.2, return_sequences=True))
→ Bidirectional(LSTM(64, dropout=0.3))
→ BatchNormalization()
→ Dense(64, activation='relu')
→ Dropout(0.4)
→ Dense(5, activation='softmax')
```

**Training Config:**
- Optimizer: Adam (lr=1e-3)
- Loss: Categorical Cross-Entropy
- Callbacks: EarlyStopping (patience=5), ReduceLROnPlateau, ModelCheckpoint
- Class Weights: Computed from class distribution to handle imbalance

---

## Module 3: BERT Model (`src/models/bert/`)

```python
# Model Architecture (HuggingFace)
Base: bert-base-uncased (110M parameters)
→ BertModel (pretrained weights)
→ Dropout(0.3)
→ Linear(hidden_size=768, 5)  # classification head
→ Softmax
```

**Training Config:**
- Trainer: HuggingFace `Trainer` API
- Learning Rate: 2e-5 with linear warmup
- Batch Size: 16 (train), 32 (eval)
- Epochs: 5 with EarlyStopping (patience=2)
- Mixed Precision: FP16 (GPU) / FP32 (CPU)

---

## Module 4: Detection Engine (`src/detection/`)

```python
# Weighted Ensemble
ensemble_probs = 0.40 * bilstm_probs + 0.60 * bert_probs

# Rule-Based Boost
if keywords_matched:
    ensemble_probs[matched_emotion] += 0.15
    ensemble_probs = ensemble_probs / ensemble_probs.sum()  # renormalize

# Final Prediction
final_emotion = EMOTION_LABELS[np.argmax(ensemble_probs)]
final_confidence = np.max(ensemble_probs)

# Mixed Emotion Check
gap = ensemble_probs[0] - ensemble_probs[1]  # top2 gap
is_mixed = gap < 0.15
```

---

## Module 5: Gemini Guidance Generator (`src/gemini/`)

**Prompt Template Structure:**
```
You are an empathetic AI learning assistant.
A student who is feeling {EMOTION} (confidence: {CONFIDENCE}%) described:
"{STUDENT_TEXT}"

Provide a structured response with exactly these 5 sections:
1. Emotional Acknowledgement
2. Targeted Learning Resources
3. Recommended Study Techniques
4. Break and Recovery Suggestions
5. Words of Encouragement
```

**Retry Logic:** Exponential backoff (1s → 2s → 4s) on API rate limit errors

---

## Module 6: Streamlit App (`app/`)

| Page | Key Components |
|---|---|
| Home (Demo) | Text area, Detect button, Emotion card, Confidence bar chart, Guidance panel |
| Analytics | 4 Plotly charts: Emotion distribution, Confidence trend, Session frequency, Model agreement |
| Model Comparison | BiLSTM vs BERT architecture cards, performance metrics |
| About | Project description, usage guide, tech stack |

**CSS Theme:** Dark glassmorphism — `#0F172A` background, `#6C63FF` accent, `Inter` font, backdrop blur cards

---

## Module 7: Analytics & Logging (`src/analytics/`)

```python
# CSV Log Row
{
    "timestamp": "2026-07-19T10:30:00",
    "session_id": "sess_abc123",
    "final_emotion": "frustrated",
    "final_confidence": 0.87,
    "is_mixed": False,
    "models_agree": True,
    "rule_applied": True,
    "keywords_matched": {"frustrated": ["stuck", "can't"]},
    "guidance_excerpt": "I understand you're feeling frustrated..."
}
```

---

## GitHub Repository

**URL:** [https://github.com/Harsh-maker007/Smartbridge_project1](https://github.com/Harsh-maker007/Smartbridge_project1)

| Stat | Value |
|---|---|
| Total Files | 71+ |
| Total Lines of Code | ~5500 |
| Test Coverage | 25+ unit tests |
| Python Modules | 20+ |
| Commits | 6+ |
