import streamlit as st

st.set_page_config(page_title="Wine Quality ML", page_icon="🍷", layout="wide")

st.title("🍷 Wine Quality ML — End‑to‑End")
st.write("""
This app lets you explore the Wine Quality dataset, train models (classification or regression),
make predictions, and review performance. Use the sidebar to navigate.
""")

st.info("Suggested flow: **Data Explorer → Visualizations → Train Models → Predict → Model Performance**")
