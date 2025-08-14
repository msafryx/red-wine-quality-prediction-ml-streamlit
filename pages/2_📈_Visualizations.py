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

# ----------------- Quality distribution -----------------
st.subheader("Quality distribution")
st.markdown("""
This histogram shows the distribution of wine quality scores in the dataset. It helps identify how many wines fall into each quality category (0–10 scale). In our binary classification, scores ≥ 7 are considered **Good**.
""")
st.plotly_chart(px.histogram(df, x="quality", nbins=10))

# ----------------- Alcohol vs Quality -----------------
st.subheader("Alcohol vs Quality")
st.markdown("""
This scatter plot displays the relationship between **alcohol content** and **quality rating**. Each point is colored based on the binary target (`Good` or `Not Good`). A visible trend here may indicate whether alcohol percentage is a strong predictor of wine quality.
""")
st.plotly_chart(px.scatter(df, x="alcohol", y="quality", color="target_good"))

# ----------------- Correlation heatmap -----------------
st.subheader("Correlation heatmap")
st.markdown("""
This heatmap visualizes the pairwise correlations between all numeric features in the dataset. A correlation close to **+1** means a strong positive relationship, while a value close to **-1** means a strong negative relationship. It’s useful for spotting potential multicollinearity and key features influencing wine quality.
""")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
st.pyplot(fig)

# ----------------- Boxplot -----------------
st.subheader("Feature-wise distribution across quality levels")
st.markdown("""
The boxplot shows the distribution of a chosen feature across different wine quality ratings. It highlights the median, quartiles, and outliers, helping to see how this feature varies between quality groups.
""")
feat_sel = st.selectbox("Select feature for boxplot", [c for c in df.columns if c != "quality"])
st.plotly_chart(px.box(df, x="quality", y=feat_sel, points="outliers"))
