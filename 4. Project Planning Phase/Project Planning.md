# Project Planning

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 3 Marks |

---

## Project Timeline (Gantt View)

| Phase | Module | Start Date | End Date | Duration | Status |
|---|---|---|---|---|---|
| 1 | Environment Setup & Config | 01 June 2026 | 03 June 2026 | 3 days | ✅ Done |
| 2 | Data Acquisition & Preprocessing | 04 June 2026 | 08 June 2026 | 5 days | ✅ Done |
| 3 | BiLSTM Model Development | 09 June 2026 | 14 June 2026 | 6 days | ✅ Done |
| 4 | BERT Model Development | 15 June 2026 | 21 June 2026 | 7 days | ✅ Done |
| 5 | Detection Engine (Ensemble + Rules) | 22 June 2026 | 25 June 2026 | 4 days | ✅ Done |
| 6 | Gemini Integration | 26 June 2026 | 28 June 2026 | 3 days | ✅ Done |
| 7 | Streamlit Web Application | 29 June 2026 | 05 July 2026 | 7 days | ✅ Done |
| 8 | Analytics & Logging | 06 July 2026 | 08 July 2026 | 3 days | ✅ Done |
| 9 | Testing | 09 July 2026 | 11 July 2026 | 3 days | ✅ Done |
| 10 | Deployment (Streamlit Cloud) | 12 July 2026 | 12 July 2026 | 1 day | ✅ Done |
| 11 | Documentation & Submission Prep | 13 July 2026 | 19 July 2026 | 7 days | ✅ Done |

**Total Duration:** ~49 days

---

## Milestone Plan

| Milestone | Target Date | Deliverable | Status |
|---|---|---|---|
| M1 — Data Ready | 08 June 2026 | Cleaned dataset, train/val/test splits | ✅ |
| M2 — Models Trained | 21 June 2026 | BiLSTM + BERT weights saved | ✅ |
| M3 — Detection Engine Live | 25 June 2026 | Unified detect() interface working | ✅ |
| M4 — Guidance Working | 28 June 2026 | Gemini returns 5-section response | ✅ |
| M5 — App Running Locally | 05 July 2026 | Full Streamlit app runs end-to-end | ✅ |
| M6 — Deployed to Cloud | 12 July 2026 | Public URL live on Streamlit Cloud | ✅ |
| M7 — Documentation Done | 19 July 2026 | All 8 phases documented for submission | ✅ |

---

## Resource Allocation

| Resource | Allocation | Tools Used |
|---|---|---|
| Developer | 4 Members (Harsh, Bhaargav, Manikanta, Praneeth) | Python, VS Code, GitHub |
| Compute (Training) | Local GPU / CPU | TensorFlow, PyTorch |
| API Credits | Google Gemini Free Tier | 1500 req/day |
| Deployment | Streamlit Community Cloud | Free |
| Dataset | HuggingFace Hub | dair-ai/emotion (16K samples) |

---

## Risk Register

| S.No | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Gemini API rate limit hit | Medium | High | Implement retry with exponential backoff; add fallback message |
| 2 | BiLSTM underfitting on small dataset | Medium | Medium | Class weighting, early stopping, data augmentation |
| 3 | BERT OOM on low-RAM machine | Low | High | Use batch size 8, gradient accumulation, CPU fallback |
| 4 | Streamlit Cloud cold start delay | High | Low | Inform users via loading spinner; use session state caching |
| 5 | Static PDF templates not accepted | Medium | High | Generate PDFs from Markdown using VS Code/Pandoc |

---

## Communication Plan

| Activity | Frequency | Channel |
|---|---|---|
| Progress tracking | Daily | GitHub commit log |
| Milestone review | Weekly | Self-review checklist |
| Issue tracking | As needed | GitHub Issues |
| Submission check | Final | Course portal |
