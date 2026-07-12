# 🧠 AI-Driven Emotion Detection & Personalized Learning Support Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Google-Gemini_API-4285F4?style=for-the-badge&logo=google&logoColor=white)

**Detect student emotions from free-text and deliver personalized AI-powered learning guidance.**

</div>

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Emotion Mapping](#-emotion-mapping)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Module Guide](#-module-guide)
- [Training the Models](#-training-the-models)
- [Running the App](#-running-the-app)
- [API Keys Setup](#-api-keys-setup)
- [Development Roadmap](#-development-roadmap)

---

## 🎯 Project Overview

This platform accepts a student's free-text description of a study problem, detects their emotional state using two AI models (BiLSTM + BERT), enhances predictions with rule-based keyword matching, and generates personalized learning guidance using the Google Gemini API.

### Detected Emotions
| Emotion | Emoji | Description |
|---|---|---|
| **Bored** | 😴 | Disengaged, uninterested, unmotivated |
| **Confident** | 😊 | Assured, understands the material |
| **Confused** | 😕 | Uncertain, lost, needs clarification |
| **Curious** | 🤔 | Engaged, wants to explore further |
| **Frustrated** | 😤 | Stressed, stuck, overwhelmed |

---

## ✨ Features

- 🤖 **Dual-Model Prediction** — BiLSTM (TensorFlow/Keras) + BERT (HuggingFace/PyTorch)
- 🎯 **Rule-Based Enhancement** — Keyword matching to boost model confidence
- 🔀 **Mixed Emotion Detection** — Identifies ambiguous emotional states
- 📊 **Confidence Scores** — Per-emotion probability breakdown for both models
- ⚖️ **Model Comparison** — Side-by-side BiLSTM vs BERT performance metrics
- 🤖 **Gemini AI Guidance** — Personalized learning advice tailored to the emotion
- 📋 **CSV Interaction Logging** — Every session is logged for later analysis
- 📈 **Analytics Dashboard** — Emotion trends, model agreement, session history
- 🌐 **Streamlit Web App** — Clean, modern, interactive UI

---

## 🏗️ Architecture

```
Student Input (Text)
        │
        ▼
  Text Preprocessing
  (clean → tokenize)
        │
   ┌────┴────┐
   ▼         ▼
BiLSTM     BERT
Model      Model
   │         │
   └────┬────┘
        │
  Rule-Based Boost
  (keyword matching)
        │
  Ensemble + Mixed
  Emotion Detection
        │
  Gemini API → Guidance
        │
  Streamlit Display + CSV Log
```

---

## 🗺️ Emotion Mapping

Since we use the `dair-ai/emotion` dataset (6 general emotions), we remap to 5 academic learning emotions:

| dair-ai Label | Academic Emotion | Reasoning |
|---|---|---|
| `sadness` | Bored | Low energy, disengagement |
| `joy` | Confident | Positive, accomplished feeling |
| `love` | Confident | Merged — positive, engaged feeling |
| `anger` | Frustrated | Blocked, irritated state |
| `fear` | Confused | Uncertainty, feeling lost |
| `surprise` | Curious | Wonder, interest, discovery |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| BiLSTM Model | TensorFlow 2.15 / Keras |
| BERT Model | PyTorch 2.2 + HuggingFace Transformers |
| Pretrained BERT | `bert-base-uncased` |
| Generative AI | Google Gemini 1.5 Flash |
| Web App | Streamlit 1.32 |
| Data Processing | Pandas, NumPy, scikit-learn |
| NLP | NLTK, spaCy, contractions |
| Visualization | Plotly, Matplotlib, Seaborn |
| Config | python-dotenv |
| Testing | pytest |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git
- A Google Gemini API key ([get one here](https://aistudio.google.com/))

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Smartbridge_project1.git
cd Smartbridge_project1
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### 5. Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 6. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 7. Download & Preprocess Dataset
```bash
python scripts/download_data.py
```

### 8. Train Models
```bash
# Train BiLSTM
python scripts/train_bilstm.py

# Train BERT (requires GPU for reasonable speed)
python scripts/train_bert.py
```

### 9. Launch the App
```bash
streamlit run app/main.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
├── config/                    # Configuration files
│   ├── settings.py            # Paths, hyperparameters, API config
│   └── emotions.py            # Emotion labels, keywords, colors
├── data/
│   ├── raw/                   # Raw dataset (gitignored)
│   ├── processed/             # Tokenized, encoded, split data
│   └── interactions/          # Session interaction logs
├── src/
│   ├── preprocessing/         # Text cleaning & tokenization
│   ├── models/
│   │   ├── bilstm/            # BiLSTM model (TF/Keras)
│   │   └── bert/              # BERT model (PyTorch/HuggingFace)
│   ├── detection/             # Inference pipeline & ensemble
│   ├── gemini/                # Gemini API integration
│   ├── analytics/             # CSV logging & chart generation
│   └── utils/                 # Shared utilities
├── models/
│   ├── bilstm/                # Saved BiLSTM model files
│   └── bert/                  # Saved BERT model files
├── app/
│   ├── main.py                # Streamlit entry point
│   ├── pages/                 # Individual app pages
│   ├── components/            # Reusable UI components
│   └── assets/                # CSS, images
├── tests/                     # pytest test suite
├── scripts/                   # CLI scripts for data/training
├── notebooks/                 # Jupyter exploration notebooks
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📦 Module Guide

| Module | Location | Description |
|---|---|---|
| **M1** Environment | `config/`, root | Setup & configuration |
| **M2** Preprocessing | `src/preprocessing/` | Data cleaning & tokenization |
| **M3** BiLSTM | `src/models/bilstm/` | BiLSTM classifier |
| **M4** BERT | `src/models/bert/` | BERT fine-tuner |
| **M5** Detection | `src/detection/` | Inference pipeline |
| **M6** Gemini | `src/gemini/` | AI guidance generation |
| **M7** Streamlit | `app/` | Web application |
| **M8** Analytics | `src/analytics/` | Logging & dashboards |
| **M9** Testing | `tests/` | Test suite |

---

## 🎓 Training the Models

### BiLSTM
```bash
python scripts/train_bilstm.py
# Model saved to: models/bilstm/bilstm_model.h5
# History saved to: models/bilstm/training_history.json
```

### BERT
```bash
python scripts/train_bert.py
# Model saved to: models/bert/model/
# Metrics saved to: models/bert/training_metrics.json
```

### Evaluate Both
```bash
python scripts/evaluate_models.py
```

---

## 🌐 Running the App

```bash
streamlit run app/main.py
```

| Page | Description |
|---|---|
| **🏠 Home** | Detect emotions, get Gemini guidance |
| **📊 Analytics** | Emotion trends and session history |
| **⚖️ Model Comparison** | BiLSTM vs BERT performance metrics |
| **ℹ️ About** | Project info and architecture |

---

## 🔑 API Keys Setup

### Google Gemini API
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Create a new API key
3. Add it to your `.env` file:
   ```
   GEMINI_API_KEY=your_key_here
   ```

---

## 🗓️ Development Roadmap

- [x] **M1** — Environment Setup & Configuration
- [ ] **M2** — Data Preprocessing Pipeline
- [ ] **M3** — BiLSTM Model Training
- [ ] **M4** — BERT Model Fine-Tuning
- [ ] **M5** — Emotion Detection Engine
- [ ] **M6** — Gemini API Integration
- [ ] **M7** — Streamlit Web Application
- [ ] **M8** — Analytics & Logging
- [ ] **M9** — Testing & Optimization

---

## 📄 License

This project is developed as part of the Smartbridge AI/ML program.

---

<div align="center">
Built with ❤️ using Python, TensorFlow, PyTorch, HuggingFace, and Google Gemini
</div>
