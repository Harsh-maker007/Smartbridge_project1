# Solution Requirements

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 3 Marks |

---

## 1. Functional Requirements

| S.No | Requirement ID | Requirement Description | Priority | Status |
|---|---|---|---|---|
| 1 | FR-01 | System shall accept free-text input (min 10 chars, max 2000 chars) from students describing their study problem | Must Have | Done |
| 2 | FR-02 | System shall classify the input into one of 5 emotions: Bored, Confident, Confused, Curious, Frustrated | Must Have | Done |
| 3 | FR-03 | System shall display the detected emotion with confidence score (%) | Must Have | Done |
| 4 | FR-04 | System shall display per-emotion probability scores as a bar chart | Must Have | Done |
| 5 | FR-05 | System shall detect mixed emotions when probability gap between top 2 emotions < 15% | Should Have | Done |
| 6 | FR-06 | System shall apply rule-based keyword matching to boost detection confidence | Should Have | Done |
| 7 | FR-07 | System shall generate 5-section personalized guidance using Google Gemini API | Must Have | Done |
| 8 | FR-08 | System shall log every interaction to a CSV file with timestamp, emotion, confidence, and session ID | Should Have | Done |
| 9 | FR-09 | System shall display an analytics dashboard with emotion distribution, session trends, and model stats | Could Have | Done |
| 10 | FR-10 | System shall display a model comparison page (BiLSTM vs BERT architecture details) | Could Have | Done |

---

## 2. Non-Functional Requirements

| S.No | Requirement ID | Description | Target |
|---|---|---|---|
| 1 | NFR-01 | **Performance** — Emotion detection response time | < 5 seconds |
| 2 | NFR-02 | **Accuracy** — Validation set emotion classification accuracy | ≥ 80% |
| 3 | NFR-03 | **Availability** — App uptime on Streamlit Cloud | 99% (free tier) |
| 4 | NFR-04 | **Usability** — No login or technical knowledge required | Zero friction |
| 5 | NFR-05 | **Scalability** — App must handle concurrent users | ≥ 10 simultaneous users |
| 6 | NFR-06 | **Security** — Gemini API key never exposed to client | Stored in Streamlit Secrets |
| 7 | NFR-07 | **Compatibility** — Works on Chrome, Firefox, Safari, Edge | All major browsers |
| 8 | NFR-08 | **Maintainability** — Modular codebase with clear folder structure | PEP-8 compliant |

---

## 3. System Constraints

| Constraint | Description |
|---|---|
| Deployment | Streamlit Community Cloud (free tier) — no TensorFlow/PyTorch on cloud |
| API Limit | Google Gemini free tier: 1500 requests/day |
| Model Serving | Cloud mode uses Gemini for detection (no local model weights) |
| Data Privacy | No user data stored beyond session-scoped CSV on the server |
| Language | English only (input language constraint) |

---

## 4. User Stories

| ID | As a... | I want to... | So that... |
|---|---|---|---|
| US-01 | Student | Type my study problem | I can get an accurate emotion detection |
| US-02 | Student | See my detected emotion with confidence | I understand how the system interpreted my text |
| US-03 | Student | Get personalized AI guidance | I can overcome my learning struggle |
| US-04 | Student | View session history | I can track my emotional patterns over time |
| US-05 | Educator | View analytics dashboard | I understand overall student emotional trends |
| US-06 | Developer | Train custom models | I can improve detection accuracy on domain-specific data |
