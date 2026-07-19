# Project Demo Planning

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 1 Mark |

---

## Demo Plan

A well-structured demo plan ensures that the team presents the project effectively, covering all key aspects in a clear and organized manner.

| S.No | Demo Section | Description | Duration (mins) | Responsible Member |
|---|---|---|---|---|
| 1 | Introduction & Problem Statement | Explain the problem: students struggle silently with emotions; generic help doesn't work | 2 | Harsh |
| 2 | Solution Overview | Introduce Emotion Detector — dual ML models + Gemini AI + Streamlit | 2 | Harsh |
| 3 | Live Feature Demo — Frustrated Input | Type "I've been stuck on pointers for 3 hours and want to give up" → show emotion detection | 3 | Harsh |
| 4 | Live Feature Demo — Curious Input | Type "I'm fascinated by how backpropagation works, want to explore more" → show Curious result | 2 | Harsh |
| 5 | Mixed Emotion Demo | Type ambiguous text → show mixed emotion flag | 1 | Harsh |
| 6 | Gemini Guidance Panel | Walk through the 5-section personalized guidance output | 2 | Harsh |
| 7 | Analytics Dashboard | Show emotion distribution chart, session history, confidence trends | 2 | Harsh |
| 8 | Model Comparison Page | Show BiLSTM vs BERT architecture cards | 1 | Harsh |
| 9 | GitHub Repository Walkthrough | Show folder structure, key modules, test suite | 2 | Harsh |
| 10 | Q&A | Answer evaluator questions | 3 | Harsh |

**Total Demo Duration: ~20 minutes**

---

## Demo Flow Summary

| Step | Activity | Notes |
|---|---|---|
| 1 | Introduction & Problem Statement | Use Empathy Map to set context — student persona (Arjun, frustrated at midnight) |
| 2 | Solution Overview | Show architecture diagram from Phase 3 |
| 3 | Live Feature Demo | Open live Streamlit app URL; demonstrate 2-3 emotion detections |
| 4 | Technical Deep Dive | Show key code modules (ensemble, Gemini client, prompt builder) |
| 5 | Testing Evidence | Run `pytest tests/` in terminal; show passing results |
| 6 | Future Roadmap | Present Scalability & Future Plan document |
| 7 | Q&A | Be ready to explain model weights, dataset, and API integration |

---

## Demo Prerequisites

| Item | Status |
|---|---|
| Streamlit Cloud app is live and accessible | ✅ |
| Gemini API key entered and working | ✅ |
| GitHub repository is public | ✅ |
| Local environment set up (pytest, streamlit) | ✅ |
| Internet connection for Gemini API calls | Required |
| Sample test inputs prepared | ✅ |

---

## Sample Demo Inputs

| Input Text | Expected Emotion |
|---|---|
| "I've been staring at this recursion problem for 3 hours and I still don't get it. I want to give up." | Frustrated |
| "I'm not sure how transformers work. The attention mechanism is unclear to me." | Confused |
| "I find it fascinating how neural networks can learn patterns from raw data!" | Curious |
| "This assignment is so boring. I've done this 10 times already." | Bored |
| "I finally understand gradient descent! It all makes sense now." | Confident |
