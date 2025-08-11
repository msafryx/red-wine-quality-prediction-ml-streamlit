import streamlit as st
import pandas as pd
from pathlib import Path

st.title("📊 Data Explorer")

DATA_PATH = Path("data/winequality-red.csv")
df = pd.read_csv(DATA_PATH)

st.write("### Overview")
st.dataframe(df.describe().T)

# Filters
flt_cols = ["alcohol", "volatile acidity", "citric acid", "sulphates", "pH"]
sliders = {}
cols_ui = st.columns(len(flt_cols))
for i, col in enumerate(flt_cols):
    min_val, max_val = float(df[col].min()), float(df[col].max())
    sliders[col] = cols_ui[i].slider(col, min_val, max_val, (min_val, max_val))

q_choices = sorted(df["quality"].unique().tolist())
q_sel = st.multiselect("Quality scores", q_choices, default=q_choices)

fdf = df.copy()
for col, (lo, hi) in sliders.items():
    fdf = fdf[fdf[col].between(lo, hi)]
fdf = fdf[fdf["quality"].isin(q_sel)]

st.success(f"Filtered rows: {len(fdf)}")
st.dataframe(fdf, use_container_width=True)
