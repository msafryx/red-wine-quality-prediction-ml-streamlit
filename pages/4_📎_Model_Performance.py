import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import plotly.graph_objects as go
from pathlib import Path

st.title("📎 Model Performance")

# Load
MODEL_PATH = Path("model.pkl")
model = joblib.load(MODEL_PATH)
feature_order = joblib.load("feature_order.pkl") if Path("feature_order.pkl").exists() else None
DATA_PATH = Path("data/winequality-red.csv")
df = pd.read_csv(DATA_PATH)
df["target_good"] = (df["quality"] >= 7).astype(int)

X = df.drop(columns=["quality", "target_good"])
if feature_order:
    X = X[feature_order]
y = df["target_good"]

# Predict
probs = model.predict_proba(X)[:,1]
preds = (probs >= 0.5).astype(int)

# Metrics
acc, f1, auc = accuracy_score(y, preds), f1_score(y, preds), roc_auc_score(y, probs)
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{acc:.3f}")
c2.metric("F1", f"{f1:.3f}")
c3.metric("ROC-AUC", f"{auc:.3f}")

# Confusion matrix
cm = confusion_matrix(y, preds)
fig_cm = go.Figure(data=go.Heatmap(
    z=cm, x=["Pred 0","Pred 1"], y=["Actual 0","Actual 1"],
    text=cm, texttemplate="%{text}", colorscale="Blues"))
st.plotly_chart(fig_cm)

# Report
with st.expander("Classification Report"):
    st.text(classification_report(y, preds, digits=3))
