# Brainstorming & Idea Prioritization Template

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 3 Marks |

---

## Step 1: Brainstorm and Idea Listing

Each team member contributed ideas related to the problem of students struggling to express their academic challenges effectively.

| S.No | Idea Title | Brief Description | Proposed By |
|---|---|---|---|
| 1 | Emotion-Aware Chatbot | A chatbot that senses frustration in a student's message and adapts its response tone | Harsh |
| 2 | Facial Expression Detector | Use webcam to detect emotions via facial expressions during online learning | Harsh |
| 3 | Free-Text Emotion Classifier | Analyze student's written description of their study problem to classify emotional state | Harsh |
| 4 | Voice Tone Analysis | Detect emotions from voice recordings submitted by students | Harsh |
| 5 | Adaptive Quiz System | Adjust quiz difficulty based on detected student confidence level | Harsh |
| 6 | Personalized Study Planner | Generate custom study plans based on detected emotions and learning gaps | Harsh |

---

## Step 2: Idea Prioritization Matrix

Rate each idea on Impact (1–5) and Feasibility (1–5). Priority Score = Impact × Feasibility.

| S.No | Idea | Impact (1–5) | Feasibility (1–5) | Priority Score | Rank |
|---|---|---|---|---|---|
| 1 | Emotion-Aware Chatbot | 4 | 3 | 12 | 3 |
| 2 | Facial Expression Detector | 4 | 2 | 8 | 5 |
| 3 | **Free-Text Emotion Classifier** | **5** | **5** | **25** | **1** |
| 4 | Voice Tone Analysis | 4 | 2 | 8 | 5 |
| 5 | Adaptive Quiz System | 4 | 3 | 12 | 3 |
| 6 | Personalized Study Planner | 5 | 4 | 20 | 2 |

---

## Step 3: Selected Idea

**Selected Idea:** Free-Text Emotion Classifier with AI-Driven Personalized Learning Guidance

**Justification:**
- Highest priority score (25/25) — maximum impact with full technical feasibility
- Requires no special hardware (no camera/microphone needed)
- Directly addresses the core problem: students can describe their study difficulties in plain text
- Enables real-time, personalized AI guidance using Google Gemini API
- Supports 5 emotion classes: Bored, Confident, Confused, Curious, Frustrated
- Can be deployed as a free Streamlit web application accessible from any device

---

## Step 4: Refined Idea Statement

> **"An AI-powered platform that accepts a student's free-text description of a study problem, detects their emotional state using a dual BiLSTM + BERT ensemble model enhanced by rule-based keyword matching, and generates personalized learning guidance using the Google Gemini API — all presented through a dark-themed, interactive Streamlit web application."**
