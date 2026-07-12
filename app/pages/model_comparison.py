"""
app/pages/model_comparison.py
================================
Model comparison page — side-by-side BiLSTM vs BERT metrics,
confusion matrices, and live comparison on custom input.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import streamlit as st
import plotly.graph_objects as go

from config.settings import BILSTM_MODEL_DIR, BERT_MODEL_DIR, EMOTION_LABELS
from config.emotions import CHART_COLORS, ORDERED_LABELS


def render_comparison():
    """Render the model comparison page."""

    st.markdown(
        """
        <div style='padding: 1rem 0;'>
            <h1 style='font-size:2rem; font-weight:700; color:#E2E8F0; margin:0;'>
                ⚖️ Model Comparison
            </h1>
            <p style='color:#94A3B8; margin-top:0.5rem;'>
                BiLSTM (TensorFlow) vs BERT (HuggingFace) — Performance Analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Load Saved Metrics ────────────────────────────────────
    bilstm_metrics = _load_metrics(BILSTM_MODEL_DIR / "test_metrics.json", "BiLSTM")
    bert_metrics   = _load_metrics(BERT_MODEL_DIR / "test_metrics.json",   "BERT")

    if bilstm_metrics or bert_metrics:
        _render_metrics_table(bilstm_metrics, bert_metrics)
        _render_metrics_bar_chart(bilstm_metrics, bert_metrics)

        if bilstm_metrics and bert_metrics:
            _render_per_class_comparison(bilstm_metrics, bert_metrics)
    else:
        st.info(
            "📌 **No trained models found yet.**\n\n"
            "Train both models first:\n"
            "```bash\n"
            "python scripts/train_bilstm.py\n"
            "python scripts/train_bert.py\n"
            "```"
        )

    # ── Architecture Overview ─────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🏛️ Architecture Overview</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="emotion-card model-badge-bilstm" style="padding:1.5rem;">
                <h3 style="color:#60A5FA; margin:0 0 1rem 0;">🔵 BiLSTM Model</h3>
                <ul style="color:#CBD5E0; line-height:1.8; padding-left:1.2rem;">
                    <li>Framework: TensorFlow 2.15 / Keras</li>
                    <li>Embedding: 128-dim word embeddings</li>
                    <li>Layers: 2× Bidirectional LSTM</li>
                    <li>Units: 128 → 64 (BiLSTM) → 64 (Dense)</li>
                    <li>Regularization: SpatialDropout + Dropout</li>
                    <li>Output: Softmax (5 classes)</li>
                    <li>Training: ~20 epochs, EarlyStopping</li>
                    <li><strong>Speed:</strong> ⚡ Fast inference</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="emotion-card model-badge-bert" style="padding:1.5rem;">
                <h3 style="color:#34D399; margin:0 0 1rem 0;">🟢 BERT Model</h3>
                <ul style="color:#CBD5E0; line-height:1.8; padding-left:1.2rem;">
                    <li>Framework: PyTorch + HuggingFace</li>
                    <li>Base: bert-base-uncased (110M params)</li>
                    <li>Fine-tuning: 4 epochs</li>
                    <li>Tokenizer: WordPiece (30K vocab)</li>
                    <li>Max Length: 128 tokens</li>
                    <li>Learning Rate: 2e-5 with warmup</li>
                    <li>Output: Linear classifier (5 classes)</li>
                    <li><strong>Accuracy:</strong> 🏆 Higher accuracy</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Ensemble Explanation ──────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🔀 Ensemble Strategy</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="guidance-panel">
            <h4 style="color:#6C63FF;">How the Ensemble Works</h4>
            <p style="color:#CBD5E0;">
                The final emotion prediction combines both models using a <strong>weighted average</strong>:
            </p>
            <div style="text-align:center; font-size:1.1rem; padding:1rem; 
                        background:rgba(108,99,255,0.1); border-radius:8px; margin:1rem 0;">
                <strong style="color:#6C63FF;">Final Score = 0.40 × BiLSTM + 0.60 × BERT</strong>
            </div>
            <p style="color:#CBD5E0;">
                BERT receives higher weight because transformer models consistently outperform RNN-based 
                models on NLP classification tasks. The weights can be tuned in <code>.env</code>:
                <code>BILSTM_WEIGHT</code> and <code>BERT_WEIGHT</code>.
            </p>
            <p style="color:#CBD5E0;">
                After ensemble averaging, <strong>rule-based keyword matching</strong> applies a 
                ±15% boost when domain-specific emotion keywords are detected in the text.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _load_metrics(path, model_name):
    if path.exists():
        with open(path) as f:
            metrics = json.load(f)
        metrics["model"] = model_name
        return metrics
    return None


def _render_metrics_table(bilstm, bert):
    st.markdown('<p class="section-header">📊 Performance Metrics (Test Set)</p>', unsafe_allow_html=True)

    metric_names = ["accuracy", "f1_macro", "precision", "recall"]
    labels       = ["Accuracy", "F1 (Macro)", "Precision", "Recall"]

    rows = []
    for metric, label in zip(metric_names, labels):
        b_val = bilstm.get(metric, 0) if bilstm else None
        r_val = bert.get(metric, 0)   if bert   else None

        winner = ""
        if b_val is not None and r_val is not None:
            if b_val > r_val:   winner = "🔵 BiLSTM"
            elif r_val > b_val: winner = "🟢 BERT"
            else:               winner = "🤝 Tie"

        rows.append({
            "Metric":   label,
            "BiLSTM":   f"{b_val:.4f}" if b_val is not None else "N/A",
            "BERT":     f"{r_val:.4f}" if r_val is not None else "N/A",
            "Winner":   winner,
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_metrics_bar_chart(bilstm, bert):
    metrics  = ["accuracy", "f1_macro", "precision", "recall"]
    labels   = ["Accuracy", "F1 Macro", "Precision", "Recall"]

    fig = go.Figure()
    if bilstm:
        fig.add_trace(go.Bar(
            name="BiLSTM",
            x=labels,
            y=[bilstm.get(m, 0) for m in metrics],
            marker_color="#3B82F6",
            text=[f"{bilstm.get(m, 0):.3f}" for m in metrics],
            textposition="outside",
        ))
    if bert:
        fig.add_trace(go.Bar(
            name="BERT",
            x=labels,
            y=[bert.get(m, 0) for m in metrics],
            marker_color="#10B981",
            text=[f"{bert.get(m, 0):.3f}" for m in metrics],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="#1A1A2E",
        plot_bgcolor="#16213E",
        font=dict(color="#E2E8F0"),
        yaxis=dict(range=[0, 1.15], gridcolor="#2D3748", title="Score"),
        xaxis=dict(gridcolor="#2D3748"),
        legend=dict(bgcolor="#1A1A2E"),
        margin=dict(t=20, b=40, l=60, r=20),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_per_class_comparison(bilstm, bert):
    st.markdown('<p class="section-header">🔍 Per-Class F1 Comparison</p>', unsafe_allow_html=True)

    emotions    = sorted(EMOTION_LABELS)
    bilstm_f1s  = [bilstm.get("per_class", {}).get(e, {}).get("f1", 0) for e in emotions]
    bert_f1s    = [bert.get("per_class",   {}).get(e, {}).get("f1", 0) for e in emotions]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="BiLSTM", x=[e.capitalize() for e in emotions],
                          y=bilstm_f1s, marker_color="#3B82F6"))
    fig.add_trace(go.Bar(name="BERT",   x=[e.capitalize() for e in emotions],
                          y=bert_f1s,   marker_color="#10B981"))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="#1A1A2E",
        plot_bgcolor="#16213E",
        font=dict(color="#E2E8F0"),
        yaxis=dict(range=[0, 1.1], gridcolor="#2D3748", title="F1 Score"),
        margin=dict(t=20, b=40, l=60, r=20),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
