import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    average_precision_score, precision_score, recall_score
)
import numpy as np
import plotly.graph_objects as go

st.title("📎 Model Performance")

# ------------------ Fixed Dataset Threshold ------------------
st.caption("**Dataset label definition:** Wines with `quality ≥ 7` are labeled as **Good (1)**, "
           "others as **Not Good (0)**. This is fixed and does not change.")

# ------------------ Adjustable Prediction Threshold ------------------
st.sidebar.header("Prediction Probability Threshold")
th = st.sidebar.slider(
    "Classify as 'Good' when P(Good) ≥",
    0.0, 1.0, 0.50, 0.01,
    help="Adjust how the model's predicted probability is converted into a class label."
)

# ---------- Load ----------
MODEL_PATH = Path("model.pkl")
DATA_PATH  = Path("data/winequality-red.csv")
FEAT_PATH  = Path("feature_order.pkl")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
feature_order = joblib.load(FEAT_PATH) if FEAT_PATH.exists() else None

df["target_good"] = (df["quality"] >= 7).astype(int)
X = df.drop(columns=["quality", "target_good"])
if feature_order:
    X = X[[c for c in feature_order if c in X.columns]]
y = df["target_good"].values

# ---------- Predict ----------
probs = model.predict_proba(X)[:, 1]
preds = (probs >= th).astype(int)

# ---------- Headline metrics ----------
acc = accuracy_score(y, preds)
f1  = f1_score(y, preds, zero_division=0)
prec = precision_score(y, preds, zero_division=0)
rec  = recall_score(y, preds, zero_division=0)
auc  = roc_auc_score(y, probs)
prev = y.mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{acc:.3f}")
c2.metric("F1",       f"{f1:.3f}")
c3.metric("Precision",f"{prec:.3f}")
c4.metric("Recall",   f"{rec:.3f}")
st.caption(f"ROC-AUC (threshold-free): **{auc:.3f}**  •  Positive rate (dataset prevalence): **{prev:.3f}**  •  Current threshold: **{th:.2f}**")

# ---------- Confusion Matrix ----------
cm = confusion_matrix(y, preds, labels=[0,1])
fig_cm = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=["Pred 0", "Pred 1"],
        y=["Actual 0", "Actual 1"],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False
    )
)
fig_cm.update_layout(title="Confusion Matrix", margin=dict(l=20, r=20, t=50, b=20))
st.plotly_chart(fig_cm, use_container_width=True)

# ---------- ROC Curve (+ marker) ----------
fpr, tpr, roc_th = roc_curve(y, probs)
valid = ~np.isinf(roc_th)
idx_roc = np.argmin(np.abs(roc_th[valid] - th))
marker_fpr, marker_tpr = fpr[valid][idx_roc], tpr[valid][idx_roc]

roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
roc_fig.add_trace(go.Scatter(
    x=[marker_fpr], y=[marker_tpr],
    mode="markers", name=f"th={th:.2f}",
    marker=dict(size=10, symbol="circle-open")
))
roc_fig.update_layout(
    title=f"ROC Curve (AUC = {auc:.3f})",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(roc_fig, use_container_width=True)

# ---------- Precision–Recall Curve (+ marker) ----------
precision, recall, pr_th = precision_recall_curve(y, probs)
ap = average_precision_score(y, probs)
idx_pr = np.argmin(np.abs(pr_th - th)) if len(pr_th) else 0
marker_prec = precision[idx_pr + 1] if len(precision) > idx_pr + 1 else precision[-1]
marker_rec  = recall[idx_pr + 1]    if len(recall)    > idx_pr + 1 else recall[-1]

pr_fig = go.Figure()
pr_fig.add_trace(go.Scatter(
    x=recall, y=precision,
    mode="lines",
    name="PR curve",
    line=dict(width=2)
))
pr_fig.add_trace(go.Scatter(
    x=[0,1], y=[prev, prev],
    mode="lines",
    name="Baseline (pos rate)",
    line=dict(dash="dash")
))
pr_fig.add_trace(go.Scatter(
    x=[marker_rec], y=[marker_prec],
    mode="markers",
    name=f"th={th:.2f}",
    marker=dict(size=10, symbol="circle-open")
))
pr_fig.update_layout(
    title=f"Precision–Recall Curve (AP = {ap:.3f})",
    xaxis_title="Recall",
    yaxis_title="Precision",
    yaxis=dict(range=[0,1.05]),
    xaxis=dict(range=[0,1.0]),
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(pr_fig, use_container_width=True)

# ---------- Classification Report ----------
raw = classification_report(y, preds, digits=3, output_dict=True)
wanted_rows = [k for k in ["0", "1", "accuracy", "macro avg", "weighted avg"] if k in raw]
rows = []
for key in wanted_rows:
    val = raw[key]
    if isinstance(val, float):
        val = {"precision": None, "recall": None, "f1-score": val, "support": len(y)}
    rows.append({"label": key, **val})

rep = pd.DataFrame(rows)
label_map = {
    "0": "Class 0 (Not Good)",
    "1": "Class 1 (Good)",
    "accuracy": "Accuracy (overall)",
    "macro avg": "Macro avg",
    "weighted avg": "Weighted avg",
}
rep["label"] = rep["label"].map(label_map).fillna(rep["label"])
for c in ["precision", "recall", "f1-score"]:
    rep[c] = pd.to_numeric(rep[c], errors="coerce").round(3)
rep["support"] = pd.to_numeric(rep.get("support", pd.Series([None]*len(rep))), errors="coerce").astype("Int64")
rep_display = rep.fillna("—")[["label", "precision", "recall", "f1-score", "support"]]

with st.expander("📄 Classification Report (detailed)", expanded=True):
    header_vals = ["Class / Avg", "Precision", "Recall", "F1", "Support"]
    cell_vals = [
        rep_display["label"].tolist(),
        rep_display["precision"].tolist(),
        rep_display["recall"].tolist(),
        rep_display["f1-score"].tolist(),
        rep_display["support"].tolist()
    ]
    table = go.Figure(
        data=[go.Table(
            header=dict(
                values=header_vals,
                fill_color="#1f2937",
                font=dict(color="white", size=16),
                align="left"
            ),
            cells=dict(
                values=cell_vals,
                fill_color=[["#0b1220" if i % 2 == 0 else "#111827" for i in range(len(rep_display))]],
                font=dict(color="white", size=14),
                align="left",
                height=28
            )
        )]
    )
    table.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=len(rep_display) * 30 + 60
    )
    st.plotly_chart(table, use_container_width=True)

# ---------- Notes ----------
st.info(
    
    "- **Dataset label threshold** (`quality ≥ 7`) → Defines what is considered 'Good' in training.\n"
    "- **Prediction probability threshold** (slider above) → Decides how predicted probabilities are turned into labels.\n\n"
    "Metrics in this page update instantly when you change the prediction threshold."
)
