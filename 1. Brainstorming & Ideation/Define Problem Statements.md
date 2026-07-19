# Define Problem Statements

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 3 Marks |

---

## Step 1: Problem Discovery

### Observation
Students often struggle silently when learning. They find it difficult to:
- Identify their own emotional state during study sessions
- Communicate their struggles effectively to educators
- Receive timely, personalized support tailored to their mindset

### Who is Affected?
| Stakeholder | Pain Point |
|---|---|
| Students | Cannot articulate study problems; receive generic help that doesn't match their emotional state |
| Teachers | Unable to monitor emotional well-being of individual students at scale |
| EdTech Platforms | Lack personalization based on learner's current emotional and cognitive state |

---

## Step 2: Problem Statement (HMW Format)

**"How might we help students receive emotionally-aware, personalized learning support by automatically detecting their emotional state from the way they describe their study problems?"**

---

## Step 3: Root Cause Analysis (5 Whys)

| Why # | Question | Answer |
|---|---|---|
| Why 1 | Why do students struggle silently? | They don't know how to express their emotional state to educators |
| Why 2 | Why can't they express it? | Traditional learning tools don't prompt or detect emotional context |
| Why 3 | Why don't tools detect emotions? | Most EdTech focuses on content delivery, not learner affect recognition |
| Why 4 | Why is affect recognition missing? | Integrating NLP emotion detection into learning platforms is technically complex |
| Why 5 | Why is it complex? | There is no accessible, off-the-shelf system that combines emotion detection with personalized guidance generation |

---

## Step 4: Problem Statement (Formal)

> **Problem:** Students in digital learning environments lack access to an intelligent system that can detect their emotional state from natural language descriptions of their study problems and automatically generate contextually appropriate, personalized learning guidance.
>
> **Impact:** This results in reduced learning outcomes, increased dropout rates, and student burnout — particularly for self-paced learners with no immediate access to a teacher.
>
> **Proposed Direction:** Build an AI system that classifies 5 student emotions (Bored, Confident, Confused, Curious, Frustrated) from free-text input and responds with targeted guidance powered by Google Gemini.

---

## Step 5: Acceptance Criteria

| Criterion | Target |
|---|---|
| Emotion classification accuracy | ≥ 80% on validation set |
| Response generation latency | < 5 seconds per query |
| Emotions supported | 5 (Bored, Confident, Confused, Curious, Frustrated) |
| Deployment platform | Accessible via public URL (Streamlit Cloud) |
| User interface | Intuitive, no technical knowledge required |
