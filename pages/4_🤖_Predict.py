import streamlit as st
import pandas as pd
from utils import load_wine, load_artifacts

st.header("🤖 Predict")

df = load_wine()
model, meta = load_artifacts()
if model is None:
    model = st.session_state.get("best_model")
    meta = {
        "task": st.session_state.get("task", "classification"),
        "feature_columns": [c for c in df.columns if c != "quality"],
        "target": "quality"
    }

if model is None:
    st.warning("No trained model found. Train and/or save a model on the **Train Models** page.")
    st.stop()

feature_cols = meta["feature_columns"]
st.write("Enter inputs:")

user = {}
for col in feature_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        v = float(df[col].median())
        mn, mx = float(df[col].min()), float(df[col].max())
        user[col] = st.number_input(col, value=v, min_value=mn, max_value=mx, step=0.1)
    else:
        opts = sorted(df[col].dropna().unique().tolist())
        user[col] = st.selectbox(col, opts)

if st.button("Predict"):
    X_new = pd.DataFrame([user])
    pred = model.predict(X_new)[0]
    if meta["task"] == "classification":
        st.success(f"Predicted quality class: **{pred}**")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new).max()
            st.info(f"Confidence: {proba:.3f}")
    else:
        st.success(f"Predicted numeric quality: **{pred:.2f}**")
