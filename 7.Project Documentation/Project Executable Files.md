# Project Executable Files

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 3 Marks |

---

## Step 1: Submission Checklist

| S.No | Item to Submit | Submitted (Yes / No / NA) |
|---|---|---|
| 1 | Complete source code (all files and folders) | Yes |
| 2 | README / Setup Guide (instructions to run the project) | Yes |
| 3 | requirements.txt / dependency file | Yes |
| 4 | requirements_full.txt (for full ML training) | Yes |
| 5 | .env.example (API key template) | Yes |
| 6 | .gitignore | Yes |
| 7 | streamlit_app.py (cloud entry point) | Yes |
| 8 | app/main.py (local entry point) | Yes |
| 9 | config/settings.py (all configuration) | Yes |
| 10 | config/emotions.py (emotion definitions) | Yes |
| 11 | src/ (all 20+ Python modules) | Yes |
| 12 | tests/ (25+ unit tests) | Yes |
| 13 | pytest.ini (test configuration) | Yes |
| 14 | All 8 Phase documentation templates | Yes |

---

## Step 2: How to Run the Project

### Option A: Cloud Demo (No Setup Required)
```
Visit the deployed Streamlit app:
https://Harsh-maker007-Smartbridge-project1-streamlit-app.streamlit.app
```
1. Paste your Gemini API key in the input box
2. Type your study problem
3. Click "Detect Emotion"

---

### Option B: Local Run (Demo Mode)
```bash
# 1. Clone the repository
git clone https://github.com/Harsh-maker007/Smartbridge_project1.git
cd Smartbridge_project1

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate    # Windows

# 3. Install lightweight dependencies
pip install -r requirements.txt

# 4. Set your Gemini API key
copy .env.example .env
# Edit .env: add GEMINI_API_KEY=your_key_here

# 5. Run the app
streamlit run streamlit_app.py
```

---

### Option C: Full Local Run (BiLSTM + BERT Training)
```bash
# Install full ML dependencies
pip install -r requirements_full.txt

# Download and preprocess dataset
python scripts/download_data.py

# Train BiLSTM model (~10 min on GPU)
python scripts/train_bilstm.py

# Fine-tune BERT model (~30 min on GPU)
python scripts/train_bert.py

# Run full-mode app (uses trained weights)
streamlit run app/main.py
```

---

## Step 3: Key Executable Files

| File | Type | Purpose |
|---|---|---|
| `streamlit_app.py` | Entry Point | Cloud/demo mode launcher |
| `app/main.py` | Entry Point | Local full-mode launcher |
| `scripts/download_data.py` | Script | Download HuggingFace dataset |
| `scripts/train_bilstm.py` | Script | Train BiLSTM model |
| `scripts/train_bert.py` | Script | Fine-tune BERT model |
| `requirements.txt` | Config | Cloud dependencies |
| `requirements_full.txt` | Config | Full ML dependencies |
| `.env.example` | Template | API key configuration |

---

## Step 4: GitHub Repository

| Field | Value |
|---|---|
| **Repository URL** | https://github.com/Harsh-maker007/Smartbridge_project1 |
| **Branch** | main |
| **Total Commits** | 7+ |
| **Total Files** | 71+ |
| **Language** | Python 100% |
| **Deployment** | Streamlit Community Cloud (Free) |
