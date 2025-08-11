import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.title("🔮 Predict Wine Quality")

# Load model
MODEL_PATH = Path("model.pkl")
model = joblib.load(MODEL_PATH)
feature_order = joblib.load("feature_order.pkl") if Path("feature_order.pkl").exists() else None

DATA_PATH = Path("data/winequality-red.csv")
df = pd.read_csv(DATA_PATH)

# Input UI
X_cols = [c for c in df.columns if c != "quality"]
desc = df[X_cols].describe()

inputs = {}
cols_ui = st.columns(3)
for i, col in enumerate(X_cols):
    min_val, max_val = float(desc.loc["min", col]), float(desc.loc["max", col])
    mean_val = float(desc.loc["mean", col])
    step = round((max_val - min_val) / 100, 4)
    inputs[col] = cols_ui[i % 3].number_input(col, min_val, max_val, mean_val, step=step)

# Predict
if st.button("Predict"):
    X_df = pd.DataFrame([inputs])[feature_order or X_cols]
    prob = float(model.predict_proba(X_df)[:,1])
    pred = "Good" if prob >= 0.5 else "Not Good"
    st.markdown(f"### Prediction: **{pred}**")
    st.progress(prob)
    st.write(f"Probability of Good: {prob:.3f}")
