import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from utils import load_artifacts

st.header("✅ Model Performance")

model, meta = load_artifacts()
X_test = st.session_state.get("X_test")
y_test = st.session_state.get("y_test")
task = (meta["task"] if meta else st.session_state.get("task", "classification"))

if model is None or X_test is None or y_test is None:
    st.warning("Please train a model on the **Train Models** page first.")
    st.stop()

preds = model.predict(X_test)

if task == "classification":
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, preds, labels=sorted(y_test.unique()))
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=sorted(y_test.unique()), y=sorted(y_test.unique()))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Report")
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

else:
    st.subheader("Regression Metrics")
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    st.write(f"**MAE:** {mae:.3f} | **RMSE:** {rmse:.3f} | **R²:** {r2:.3f}")

    st.subheader("Residual Plot")
    residuals = y_test - preds
    fig = px.scatter(x=preds, y=residuals, labels={"x":"Predicted", "y":"Residual"})
    st.plotly_chart(fig, use_container_width=True)
