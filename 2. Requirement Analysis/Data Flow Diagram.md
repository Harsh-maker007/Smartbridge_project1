# Data Flow Diagram

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 3 Marks |

---

## Level 0 — Context Diagram

```
┌─────────────┐        Free-Text Input        ┌──────────────────────────────────┐
│             │ ─────────────────────────────► │                                  │
│   Student   │                                │   Emotion Detector Platform      │
│             │ ◄───────────────────────────── │                                  │
└─────────────┘   Detected Emotion + Guidance  └──────────────────────────────────┘
                                                            │
                                                 API Calls  │
                                                            ▼
                                               ┌─────────────────────┐
                                               │   Google Gemini API  │
                                               └─────────────────────┘
```

---

## Level 1 — High-Level Data Flow

```
Student Input (Free Text)
         │
         ▼
 ┌───────────────────┐
 │  1. Preprocessing  │  → Clean text, remove noise, tokenize
 └───────────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌──────────┐ ┌──────────┐
│  BiLSTM  │ │   BERT   │   (Parallel inference)
│  Model   │ │  Model   │
└──────────┘ └──────────┘
    │         │
    └────┬────┘
         │  Probability Vectors
         ▼
 ┌────────────────────┐
 │  2. Ensemble Engine │  → 40% BiLSTM + 60% BERT weighted average
 └────────────────────┘
         │
         ▼
 ┌────────────────────────┐
 │  3. Rule-Based Boost   │  → Keyword matching → confidence boost (+15%)
 └────────────────────────┘
         │
         ▼
 ┌────────────────────────┐
 │  4. Mixed Emotion Check │  → Probability gap threshold analysis
 └────────────────────────┘
         │
    ┌────┴────────────────┐
    │                     │
    ▼                     ▼
Final Emotion        Gemini API Call
+ Confidence        (Prompt Builder)
    │                     │
    └────────┬────────────┘
             ▼
     ┌───────────────┐
     │  5. Streamlit  │  → Display Emotion Card + Confidence Bars + Guidance
     │     UI         │
     └───────────────┘
             │
             ▼
     ┌───────────────┐
     │  6. CSV Logger │  → Save interaction to data/interactions/interaction_logs.csv
     └───────────────┘
```

---

## Level 2 — Detailed Data Flows

### 2.1 Preprocessing Flow
| Input | Process | Output |
|---|---|---|
| Raw student text | Remove HTML, URLs, mentions | Cleaned text |
| Cleaned text | Lowercase, remove punctuation | Normalized text |
| Normalized text | NLTK stopword removal (negation-preserving) | Filtered tokens |
| Filtered tokens | WordNet lemmatization | Final tokens |
| Tokens (BiLSTM path) | Integer encoding + padding (maxlen=100) | Padded sequence |
| Tokens (BERT path) | BertTokenizer (bert-base-uncased) | input_ids + attention_mask |

### 2.2 Model Inference Flow
| Model | Input | Output |
|---|---|---|
| BiLSTM | Padded integer sequence (100,) | Softmax probability vector (5,) |
| BERT | input_ids + attention_mask | Softmax probability vector (5,) |
| Ensemble | 2 probability vectors | Weighted average vector (5,) |

### 2.3 Gemini Guidance Flow
| Input | Process | Output |
|---|---|---|
| Final emotion + confidence | Prompt builder → emotion-specific template | Structured prompt |
| Structured prompt | Gemini API call (gemini-pro) | 5-section guidance text |
| Guidance text | HTML formatting | Rendered guidance panel |

---

## Data Entities

| Entity | Description | Storage |
|---|---|---|
| Student Input | Raw text describing study problem | Streamlit session state |
| Detection Result | Emotion, confidence, probabilities, keywords | Streamlit session state + CSV |
| Guidance Text | 5-section AI-generated learning support | Streamlit session state |
| Interaction Log | Full row per session | data/interactions/interaction_logs.csv |
| Model Weights | BiLSTM (.h5) and BERT (safetensors) | data/models/ (local only) |
