# Technology Stack

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 3 Marks |

---

## Technology Stack Overview

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Frontend / UI** | Streamlit | 1.32+ | Interactive web application |
| **Styling** | Custom CSS | — | Dark glassmorphism theme |
| **ML — Deep Learning** | TensorFlow / Keras | 2.15 | BiLSTM model training & inference |
| **ML — Transformers** | PyTorch | 2.2 | BERT model fine-tuning |
| **NLP — Pretrained** | HuggingFace Transformers | 4.38 | bert-base-uncased tokenizer & model |
| **NLP — Utilities** | NLTK | 3.8 | Stopword removal, lemmatization |
| **Generative AI** | Google Gemini API | gemini-pro | Personalized guidance generation |
| **Gemini SDK** | google-generativeai | 0.5+ | Python client for Gemini |
| **Data Processing** | Pandas | 2.2 | Dataframe ops, CSV logging |
| **Data Processing** | NumPy | 1.26 | Array/matrix operations |
| **Visualization** | Plotly | 5.20 | Interactive charts (emotion bars, trends) |
| **Dataset** | Hugging Face `dair-ai/emotion` | — | 6-class → remapped to 5-class |
| **Environment** | Python | 3.10+ | Core language |
| **Deployment** | Streamlit Community Cloud | Free tier | Public web deployment |
| **Version Control** | Git + GitHub | — | Source code management |
| **Configuration** | python-dotenv | 1.0 | Environment variable management |
| **Testing** | pytest | 8.1 | Unit testing |

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────┐
│                 PRESENTATION LAYER                   │
│        Streamlit Web App (app/main.py)               │
│   Home | Analytics | Model Comparison | About        │
└─────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│                  BUSINESS LOGIC LAYER                │
│   Detection Engine    │    Guidance Generator        │
│  (BiLSTM + BERT +     │   (Gemini API + Prompt       │
│   Rule-Based)         │    Builder)                  │
└─────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│                    DATA LAYER                         │
│   Preprocessing    │  Model Weights  │  CSV Logs     │
│   Pipeline         │  (H5 + Safety   │  (Pandas)     │
│   (NLTK)           │   Tensors)      │               │
└─────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│                  EXTERNAL SERVICES                   │
│         Google Gemini API  │  HuggingFace Hub        │
└─────────────────────────────────────────────────────┘
```

---

## Why This Stack?

| Decision | Rationale |
|---|---|
| Streamlit over Flask/Django | Rapid development, Python-native, free cloud deployment |
| BiLSTM + BERT ensemble | Combines sequential pattern recognition with contextual understanding |
| Gemini API for guidance | State-of-the-art LLM, generous free tier (1500 req/day), easy Python integration |
| Gemini-only on cloud | Avoids TensorFlow/PyTorch heavy dependencies on Streamlit's free tier |
| HuggingFace `dair-ai/emotion` | Publicly available, well-balanced 6-class dataset remappable to our 5 emotions |
| Plotly for charts | Interactive, aesthetically consistent with dark theme |
| CSV logging | Zero infrastructure cost, portable, human-readable |

---

## Development Tools

| Tool | Purpose |
|---|---|
| VS Code | Primary code editor |
| GitHub | Version control and repository hosting |
| Google AI Studio | Gemini API key management |
| Streamlit Cloud | Free public deployment |
| pytest | Automated unit testing |
