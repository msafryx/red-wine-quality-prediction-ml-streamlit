import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

st.title("📈 Visualizations")

DATA_PATH = Path("data/winequality-red.csv")
df = pd.read_csv(DATA_PATH)
df["target_good"] = (df["quality"] >= 7).astype(int)

# Quality distribution
st.subheader("Quality distribution")
st.plotly_chart(px.histogram(df, x="quality", nbins=10))

# Alcohol vs Quality
st.subheader("Alcohol vs Quality")
st.plotly_chart(px.scatter(df, x="alcohol", y="quality", color="target_good"))

# Correlation heatmap
st.subheader("Correlation heatmap")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
st.pyplot(fig)

# Boxplot
feat_sel = st.selectbox("Feature for boxplot", [c for c in df.columns if c != "quality"])
st.plotly_chart(px.box(df, x="quality", y=feat_sel, points="outliers"))
