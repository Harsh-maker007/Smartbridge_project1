# Scalability & Future Plan

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 1 Mark |

---

## Current System Limitations

| S.No | Limitation | Impact | Priority to Address |
|---|---|---|---|
| 1 | English-only emotion detection | Excludes non-English speaking students | High |
| 2 | No user authentication or personalization | Cannot track individual student progress over sessions | High |
| 3 | Gemini free tier limit (1500 req/day) | App unusable after daily quota exhaustion | High |
| 4 | BiLSTM + BERT weights not deployed on cloud | Full ML pipeline unavailable on Streamlit free tier | Medium |
| 5 | Single user text input only | Cannot analyze audio, images, or video for emotion | Medium |
| 6 | Static keyword dictionary | New slang or domain-specific terms may be missed | Low |
| 7 | CSV-based logging | Not scalable beyond ~100K interactions | Low |

---

## Scalability Plan

| S.No | Scalability Aspect | Current State | Proposed Upgrade / Solution |
|---|---|---|---|
| 1 | User Load | 10 concurrent Streamlit Cloud users | Migrate to FastAPI backend + React frontend on cloud VMs (AWS/GCP) |
| 2 | Data Storage | Flat CSV file | Migrate to PostgreSQL or Firebase Realtime DB for structured, indexed storage |
| 3 | Performance | Gemini API latency ~2–3s | Cache Gemini responses for identical inputs using Redis |
| 4 | Model Deployment | Gemini-only on cloud | Deploy BiLSTM + BERT as REST microservices using FastAPI + Docker on a VPS |
| 5 | Security | Gemini key in Streamlit Secrets | Use OAuth2 for user auth; store keys in AWS Secrets Manager |
| 6 | Multilingual Support | English only | Integrate Google Translate API for preprocessing; retrain on multilingual datasets |
| 7 | Analytics | Static Plotly charts | Build a real-time dashboard with WebSocket streaming (Plotly Dash / Grafana) |

---

## Future Feature Roadmap

| Priority | Feature | Description | ETA |
|---|---|---|---|
| 🔴 High | User Login & Profiles | Track individual student emotional journey across sessions | Q3 2026 |
| 🔴 High | Multilingual Support | Detect emotions from Hindi, Tamil, Spanish text inputs | Q3 2026 |
| 🟡 Medium | Voice Input | Analyze emotion from audio recordings via speech-to-text + NLP | Q4 2026 |
| 🟡 Medium | Teacher Dashboard | Aggregated class-level emotion analytics for educators | Q4 2026 |
| 🟡 Medium | Mobile App | React Native app wrapping the core detection API | Q4 2026 |
| 🟢 Low | Facial Expression Fusion | Multimodal detection combining text + webcam | 2027 |
| 🟢 Low | LMS Integration | Plugin for Moodle / Google Classroom | 2027 |
| 🟢 Low | Custom Model Fine-Tuning | Let institutions train on their own student data | 2027 |

---

## Scalability Architecture (Future State)

```
Student (Web / Mobile)
        │
        ▼
   Load Balancer (Nginx)
        │
   ┌────┴────────────────┐
   │  FastAPI Backend     │
   │  (Docker Containers) │
   │                     │
   │  ┌───────────────┐  │
   │  │ BiLSTM Service│  │
   │  │ BERT Service  │  │
   │  │ Gemini Client │  │
   │  └───────────────┘  │
   └─────────┬───────────┘
             │
   ┌─────────┴─────────┐
   │   PostgreSQL DB    │
   │   Redis Cache      │
   │   AWS S3 (Models)  │
   └───────────────────┘
```
