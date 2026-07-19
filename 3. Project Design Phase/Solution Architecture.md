# Solution Architecture

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 3 Marks |

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT CLOUD                               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    PRESENTATION LAYER                         │   │
│  │                                                              │   │
│  │  ┌──────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────┐  │   │
│  │  │  Home    │  │  Analytics  │  │    Model     │  │About │  │   │
│  │  │  Page    │  │  Dashboard  │  │  Comparison  │  │ Page │  │   │
│  │  └──────────┘  └─────────────┘  └──────────────┘  └──────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    BUSINESS LOGIC LAYER                       │   │
│  │                                                              │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │              DETECTION ENGINE                        │    │   │
│  │  │                                                     │    │   │
│  │  │  Text Input → Preprocessing → [BiLSTM] + [BERT]    │    │   │
│  │  │              → Ensemble → Rule-Boost → Final Emotion│    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │                                                              │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │          GEMINI GUIDANCE GENERATOR                   │    │   │
│  │  │  Prompt Builder → Gemini API → 5-Section Response   │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      DATA LAYER                               │   │
│  │  CSV Logs (pandas)  │  Model Weights (H5/SafeTensors)        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │ HTTPS
┌──────────────────────────────▼──────────────────────────────────────┐
│                     EXTERNAL SERVICES                                │
│          Google Gemini API (gemini-pro)                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Architecture

### `src/` — Core Business Logic

| Module | File | Responsibility |
|---|---|---|
| Preprocessing | `cleaner.py` | HTML removal, normalization, stopwords |
| Preprocessing | `tokenizer.py` | Tokenization for BiLSTM path |
| Preprocessing | `encoder.py` | Label encoding, integer encoding |
| Preprocessing | `pipeline.py` | Orchestrates full preprocessing flow |
| BiLSTM | `architecture.py` | Keras model definition |
| BiLSTM | `trainer.py` | Training loop with callbacks |
| BiLSTM | `inference.py` | Load weights + predict |
| BERT | `architecture.py` | HuggingFace model definition |
| BERT | `trainer.py` | HuggingFace Trainer fine-tuning |
| BERT | `inference.py` | Tokenize + forward pass + predict |
| Detection | `rule_based.py` | Keyword dictionary + confidence boost |
| Detection | `ensemble.py` | Weighted average of model outputs |
| Detection | `mixed_emotion.py` | Gap threshold for mixed detection |
| Detection | `detector.py` | Unified detection interface (singleton) |
| Detection | `gemini_detector.py` | Cloud-mode: Gemini as emotion classifier |
| Gemini | `client.py` | API client with retry + backoff |
| Gemini | `prompt_builder.py` | Emotion-specific prompt templates |
| Gemini | `guidance_generator.py` | Orchestrates Gemini call + formatting |
| Analytics | `logger.py` | CSV interaction logging |
| Analytics | `aggregator.py` | KPI computation from logs |
| Analytics | `charts.py` | 5 Plotly chart generators |

---

## Deployment Architecture

| Environment | Mode | Models Used | Entry Point |
|---|---|---|---|
| **Local (Full)** | Full ML Pipeline | BiLSTM + BERT + Gemini | `app/main.py` |
| **Streamlit Cloud** | Demo Mode | Gemini API only | `streamlit_app.py` |

---

## Data Flow Summary

```
Student Input
    → Text Preprocessing (NLTK)
    → BiLSTM Inference (TF/Keras) ──┐
                                    ├─→ Weighted Ensemble
    → BERT Inference (PyTorch)  ────┘
    → Rule-Based Keyword Boost
    → Mixed Emotion Check
    → Final Emotion + Confidence
    → Gemini Prompt Builder
    → Google Gemini API Call
    → 5-Section Guidance Response
    → CSV Logger
    → Streamlit UI Render
```

---

## Config & Settings

| File | Purpose |
|---|---|
| `config/settings.py` | All hyperparameters, paths, thresholds |
| `config/emotions.py` | Emotion labels, keywords, colors, metadata |
| `.env` | GEMINI_API_KEY (local) |
| `.streamlit/secrets.toml` | GEMINI_API_KEY (Streamlit Cloud) |
