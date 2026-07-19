# Empathy Map Canvas

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 3 Marks |

---

## Target User Persona

**Name:** Arjun Sharma  
**Role:** Undergraduate Engineering Student (2nd Year)  
**Context:** Studying Data Structures late at night, preparing for an exam tomorrow

---

## Empathy Map

### 👀 SAYS
> *"I've been staring at this recursion problem for 2 hours and I still don't get it."*  
> *"I feel so stupid compared to my classmates."*  
> *"I keep starting over but I can't make progress."*  
> *"I just want someone to explain this differently."*  
> *"Maybe engineering isn't for me."*

---

### 🤔 THINKS
- "Everyone else probably understands this easily."
- "I'm running out of time and I'm panicking."
- "Will I fail tomorrow's exam?"
- "Am I studying wrong? Is my approach broken?"
- "I wish I had a tutor right now."

---

### 😊 FEELS
| Emotion | Trigger |
|---|---|
| **Frustrated** | Hours of effort with no visible progress |
| **Confused** | Contradictory explanations from different resources |
| **Anxious** | Upcoming deadline and fear of failure |
| **Isolated** | No one available to help at midnight |
| **Demotivated** | Comparison with peers |

---

### 🏃 DOES
- Switches between YouTube videos, textbooks, and StackOverflow
- Types increasingly desperate queries into search engines
- Closes laptop and tries to sleep, then reopens it
- Texts friends but nobody responds
- Rereads the same paragraph multiple times hoping it clicks

---

## Pain Points Summary

| S.No | Pain Point | Severity |
|---|---|---|
| 1 | No system detects that he is frustrated and responds with empathy | High |
| 2 | Generic search results don't adapt to his emotional state | High |
| 3 | No personalized guidance available at odd hours | High |
| 4 | No feedback loop to know if his study approach is wrong | Medium |
| 5 | Feeling of isolation without peer/teacher support | Medium |

---

## How Our Solution Helps

| Pain Point | Solution Feature |
|---|---|
| Frustration undetected | BiLSTM + BERT + Gemini classifies emotion from his text |
| Generic responses | Gemini generates emotion-specific, personalized guidance |
| No 24/7 support | Streamlit app is always available online |
| Wrong approach | Guidance includes study strategy recommendations |
| Isolation | App responds empathetically, addressing the emotional context |
