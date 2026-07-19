# Proposed Solution

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 3 Marks |

---

## Solution Overview

The **Emotion Detector** is an AI-Driven Emotion Detection & Personalized Learning Support Platform. It bridges the gap between how students feel during learning and the kind of support they receive by using cutting-edge Natural Language Processing and Generative AI.

---

## Core Components

### Component 1: Dual-Model Emotion Classifier
A weighted ensemble of two independently trained deep learning models:

**BiLSTM (Bidirectional Long Short-Term Memory)**
- Architecture: Embedding(10000, 128) → SpatialDropout(0.3) → BiLSTM(128) × 2 → BatchNorm → Dense(5, softmax)
- Input: Integer-encoded, padded token sequence (maxlen=100)
- Strength: Captures sequential and bidirectional context in text
- Training: EarlyStopping, ReduceLROnPlateau, class-weighting for balance

**BERT (bert-base-uncased)**
- Architecture: Pretrained transformer with fine-tuned classification head
- Input: Subword tokenized text with attention masks
- Strength: Deep contextual understanding of language semantics
- Training: HuggingFace Trainer API, FP16 mixed precision

**Ensemble Strategy:** 40% BiLSTM + 60% BERT weighted probability average

---

### Component 2: Rule-Based Keyword Enhancement
A dictionary of emotion-specific keywords for each of the 5 classes:
- **Bored:** "boring", "dull", "uninteresting", "monotonous", "tedious"
- **Confident:** "understand", "got it", "clear", "mastered", "ready"
- **Confused:** "don't understand", "lost", "unclear", "confused", "makes no sense"
- **Curious:** "wonder", "interesting", "want to know", "fascinating", "how does"
- **Frustrated:** "stuck", "giving up", "can't", "impossible", "frustrated"

When keywords are matched, the corresponding emotion's probability is boosted by **+15%** and renormalized.

---

### Component 3: Mixed Emotion Detection
If the probability gap between the top-2 detected emotions is < **15%**, the system flags a "mixed emotion" state and reports both emotions to provide more nuanced guidance.

---

### Component 4: Gemini-Powered Guidance Generator
Uses Google Gemini API (gemini-pro) to generate a structured 5-section response:
1. **Emotional Acknowledgement** — Validates the student's feeling
2. **Targeted Resources** — Specific study materials for their topic
3. **Study Techniques** — Emotion-matched learning strategies
4. **Break & Recovery Suggestions** — Mindful breaks if frustrated/bored
5. **Words of Encouragement** — Motivational closing

---

### Component 5: Streamlit Web Application
A fully responsive, dark glassmorphism UI with:
- **Home Page:** Input → Detection → Guidance in one flow
- **Analytics Page:** 5 Plotly charts — emotion distribution, confidence trends, session history
- **Model Comparison Page:** BiLSTM vs BERT architecture and performance comparison
- **About Page:** Project description and usage guide

---

### Component 6: Interaction Logger & Analytics
Every interaction is logged to `data/interactions/interaction_logs.csv` with fields:
`timestamp, session_id, user_text, final_emotion, final_confidence, is_mixed, bilstm_emotion, bert_emotion, models_agree, rule_applied, guidance_excerpt`

---

## Solution Differentiators

| Feature | Generic Chatbot | Google Search | Emotion Detector |
|---|---|---|---|
| Emotion Detection | ❌ | ❌ | ✅ |
| Personalized to emotional state | ❌ | ❌ | ✅ |
| Dual AI models | ❌ | ❌ | ✅ |
| Rule-based enhancement | ❌ | ❌ | ✅ |
| Mixed emotion awareness | ❌ | ❌ | ✅ |
| Analytics dashboard | ❌ | ❌ | ✅ |
| No login required | ✅ | ✅ | ✅ |
| Free to use | ❌ | ✅ | ✅ |
