# Performance Testing

| Field | Details |
|---|---|
| **Date** | 19 July 2026 |
| **Team ID** | SB-AIML-2026-0042 |
| **Project Name** | Emotion Detector — AI-Driven Emotion Detection & Personalized Learning Support Platform |
| **Maximum Marks** | 5 Marks |

---

## Step 1: Testing Overview

| Field | Details |
|---|---|
| **Testing Tool Used** | pytest (unit tests) + manual response time measurement (Python `time` module) |
| **Type of Testing** | Functional Testing, Load Testing (manual), Unit Testing, Integration Testing |
| **Target Module / API** | Detection Engine (`src/detection/`), Gemini API Client (`src/gemini/client.py`), Streamlit UI |
| **Test Environment** | Local (Windows 11, Python 3.10, 16GB RAM) + Streamlit Community Cloud (Free Tier) |
| **Test Date** | 09 July 2026 – 12 July 2026 |

---

## Step 2: Test Scenarios

| S.No | Test Scenario | Input | Expected Output | Result |
|---|---|---|---|---|
| 1 | Valid frustrated input | "I've been stuck on this for hours and want to give up" | Emotion: Frustrated, Confidence > 70% | ✅ Pass |
| 2 | Valid curious input | "I wonder how neural networks actually learn, it's fascinating" | Emotion: Curious, Confidence > 65% | ✅ Pass |
| 3 | Short invalid input (< 10 chars) | "help" | Validation error message shown | ✅ Pass |
| 4 | Empty input | "" | Validation error message shown | ✅ Pass |
| 5 | Very long input (2000 chars) | 2000-char essay | Processed correctly, no crash | ✅ Pass |
| 6 | Mixed emotion input | "I'm a bit curious but also really confused" | Mixed emotion flag = True | ✅ Pass |
| 7 | Keyword-boosted input | "I feel so bored and this is so boring and dull" | Rule applied = True, Bored detected | ✅ Pass |
| 8 | Gemini API unavailable | Mock Gemini timeout | Fallback message displayed, no crash | ✅ Pass |
| 9 | Concurrent users (manual) | 5 browser tabs simultaneously | All respond correctly | ✅ Pass |
| 10 | CSV logging | After any detection | New row appended to interaction_logs.csv | ✅ Pass |

---

## Step 3: Unit Test Results

### Test Suite: `tests/test_preprocessing.py`

| Test | Description | Result |
|---|---|---|
| `test_remove_html` | HTML tags removed from text | ✅ Pass |
| `test_remove_urls` | URLs stripped from text | ✅ Pass |
| `test_remove_mentions_hashtags` | @mentions and #hashtags removed | ✅ Pass |
| `test_normalize_whitespace` | Multiple spaces collapsed | ✅ Pass |
| `test_clean_text_full` | Full pipeline produces non-empty output | ✅ Pass |
| `test_clean_for_bilstm` | BiLSTM path: lowercase, no stopwords | ✅ Pass |
| `test_clean_for_bert` | BERT path: preserves punctuation/case | ✅ Pass |
| `test_empty_input` | Empty string returns empty string | ✅ Pass |

### Test Suite: `tests/test_detection_pipeline.py`

| Test | Description | Result |
|---|---|---|
| `test_rule_based_no_match` | No keywords → rule not applied | ✅ Pass |
| `test_rule_based_match` | "stuck" → frustrated boost applied | ✅ Pass |
| `test_ensemble_weights` | 40/60 weights sum to 1.0 | ✅ Pass |
| `test_ensemble_output_shape` | Output shape is (5,) | ✅ Pass |
| `test_mixed_emotion_detected` | Gap < 0.15 → mixed = True | ✅ Pass |
| `test_single_emotion` | Gap > 0.15 → mixed = False | ✅ Pass |

### Test Suite: `tests/test_analytics.py`

| Test | Description | Result |
|---|---|---|
| `test_log_creates_file` | log_interaction() creates CSV | ✅ Pass |
| `test_log_appends_row` | Second call appends, not overwrites | ✅ Pass |
| `test_aggregator_kpis` | KPIs computed correctly from mock data | ✅ Pass |

---

## Step 4: Performance Metrics

| Metric | Value | Target | Status |
|---|---|---|---|
| Emotion Detection Response Time (Demo Mode) | ~2.8 seconds | < 5 seconds | ✅ Pass |
| Gemini Guidance Generation Time | ~1.5 seconds | < 5 seconds | ✅ Pass |
| End-to-End Response Time | ~4.2 seconds | < 8 seconds | ✅ Pass |
| App Cold Start (Streamlit Cloud) | ~12 seconds | < 30 seconds | ✅ Pass |
| CSV Log Write Time | < 0.1 seconds | < 1 second | ✅ Pass |
| Analytics Dashboard Load Time | < 2 seconds | < 5 seconds | ✅ Pass |

---

## Step 5: Model Performance Metrics (Training Results)

| Metric | BiLSTM | BERT | Ensemble |
|---|---|---|---|
| Training Accuracy | 84.2% | 89.7% | — |
| Validation Accuracy | 81.3% | 87.4% | 88.1% |
| Test Accuracy | 80.8% | 86.9% | **87.6%** |
| Macro F1-Score | 0.79 | 0.86 | **0.87** |
| Inference Time (per sample) | 45ms | 120ms | 165ms |

---

## Step 6: Issues Found & Resolved

| S.No | Issue | Resolution |
|---|---|---|
| 1 | `contractions` library fails on Streamlit Cloud | Removed from requirements.txt; handled inline |
| 2 | Version conflicts in requirements.txt | Relaxed pinning to `>=` for cloud deployment |
| 3 | Gemini API key not found on cloud | Added interactive UI key input field |
| 4 | Remote push rejected (diverged history) | Resolved via `git pull --rebase` |
