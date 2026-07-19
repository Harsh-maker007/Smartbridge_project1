# Communication Plan

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 1 Mark |

---

## Communication Plan

Effective communication is essential for a successful project demonstration. This document outlines the communication strategy used within the team and with stakeholders throughout the project lifecycle, including how updates, issues, and feedback were managed.

| S.No | Communication Type | Frequency | Channel / Tool | Participants | Purpose |
|---|---|---|---|---|---|
| 1 | Team Standup | Daily | GitHub Commit Log | Harsh (Solo Developer) | Track daily progress and blockers |
| 2 | Progress Update | Weekly | Self-review Checklist (Markdown task.md) | Developer + Mentor | Ensure milestone targets are on track |
| 3 | Issue / Bug Discussion | As Needed | GitHub Issues + VS Code Debugger | Developer | Identify and resolve technical bugs |
| 4 | Stakeholder Review | Bi-Weekly | WhatsApp / Email to Mentor | Developer + Course Mentor | Get feedback on design and direction |
| 5 | Final Demo Rehearsal | Once | Streamlit Cloud App (Local Test) | Developer | End-to-end demo walkthrough before submission |
| 6 | Submission Check | Once | Course Portal + GitHub | Developer + Evaluator | Final repository submission verification |

---

## Communication Challenges & Resolutions

| S.No | Challenge Faced | Resolution Applied |
|---|---|---|
| 1 | No real-time team feedback (solo project) | Used self-review checklists and AI pair programming (Antigravity) for feedback |
| 2 | Gemini API error messages unclear | Added detailed logging and retry error reporting in `src/gemini/client.py` |
| 3 | Streamlit Cloud deployment errors not visible in real-time | Used Streamlit's "Manage App" terminal for live log inspection |
| 4 | Git push rejection due to diverged remote | Resolved via `git pull --rebase origin main` followed by force-safe push |
| 5 | Uncertainty about template requirements | Analyzed reference GitHub repo (Ravi-teja-777) and Drive folder to align |

---

## Communication Tools Used

| Tool | Purpose |
|---|---|
| GitHub | Version control, commit history, issue tracking |
| VS Code | Primary development and debugging |
| Streamlit Cloud Dashboard | Deployment logs and app monitoring |
| Google AI Studio | Gemini API key management and testing |
| WhatsApp / Email | Mentor communication for milestone review |
