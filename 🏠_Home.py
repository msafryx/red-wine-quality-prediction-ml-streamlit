import streamlit as st
import pandas as pd
from pathlib import Path

# Page settings (add icon here)
st.set_page_config(
    page_title="Wine Quality (Red) – ML App",
    page_icon="🍷",  # icon for app + tab
    layout="wide"
)

# Load dataset
DATA_PATH = Path("data/winequality-red.csv")
df = pd.read_csv(DATA_PATH)

# Home Page content
st.title("🍷 Wine Quality (Red) — Machine Learning App")
st.markdown("""
Welcome to the Wine Quality Prediction App!  
This tool predicts whether a red wine is **Good** (quality ≥ 7) or **Not Good**  
based on its chemical properties.

**Use the sidebar** to:
- 📊 Explore the dataset
- 📈 View visualizations
- 🔮 Make predictions
- 📎 See model performance
""")

# Quick dataset info
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{df.shape[0]}")
c2.metric("Columns", f"{df.shape[1]}")
c3.metric("Target", "Binary: quality ≥ 7")

# Dataset preview
with st.expander("Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)
