# Problem-Solution Fit

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 3 Marks |

---

## Problem Space

### Core Problem
Students in digital learning environments experience a range of emotional states (frustration, confusion, boredom, etc.) while studying. These emotions directly impact their ability to absorb and retain knowledge. However, existing EdTech platforms treat all students identically — delivering the same content regardless of the learner's current emotional or cognitive state.

### Pain Points Validated
| Pain Point | Validation Source |
|---|---|
| Students cannot identify why they are stuck | User interviews with engineering students |
| Generic help resources don't address emotional context | Analysis of existing EdTech platforms |
| No 24/7 empathetic support available | Survey: 78% of students study after 10 PM with no peer/tutor access |
| Delayed feedback loop between student struggle and teacher awareness | Educator interviews |

---

## Solution Space

### Our Solution: Emotion Detector Platform
An AI-powered web application that:
1. Accepts student's free-text description of their study problem
2. Detects emotional state using BiLSTM + BERT ensemble + rule-based keyword engine
3. Generates personalized learning guidance using Google Gemini AI
4. Logs and visualizes emotional trends over time

---

## Problem-Solution Fit Matrix

| Customer Problem | Solution Feature | Fit Score (1–5) |
|---|---|---|
| Can't identify emotional state | Automatic emotion classification (5 classes) | 5 |
| Generic help doesn't match mood | Gemini generates emotion-specific guidance | 5 |
| No empathetic response | 5-section structured guidance with emotional acknowledgement | 5 |
| No 24/7 support | Always-online Streamlit Cloud deployment | 5 |
| No trend visibility | Analytics dashboard with session history | 4 |
| Unsure if approach is correct | Study technique recommendations in guidance | 4 |

**Overall Fit Score: 28/30 (93%)**

---

## Value Proposition

| For | Who | Our product | Is a | That | Unlike | Our solution |
|---|---|---|---|---|---|---|
| Students | who struggle silently while studying | Emotion Detector | AI-powered web app | detects emotions from text and gives tailored guidance | generic search or Q&A forums | responds with empathy and personalized strategy |

---

## Assumptions & Risks

| Assumption | Risk if Wrong | Mitigation |
|---|---|---|
| Students will describe problems in English | Lower accuracy for non-English inputs | Add language detection; document English-only constraint |
| Gemini API will remain on free tier | Cost barrier if API pricing changes | Implement local Gemini-lite fallback |
| Students willing to type full sentences | Short inputs reduce accuracy | Minimum character validation + example prompts |
| BiLSTM + BERT ensemble improves accuracy | Ensemble could underperform single model | Compare models; use best performer |
