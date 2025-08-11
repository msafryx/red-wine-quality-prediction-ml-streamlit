import streamlit as st
import plotly.express as px
import pandas as pd
from utils import load_wine

st.header("📈 Visualizations")
df = load_wine()

num_cols = df.select_dtypes(include="number").columns.tolist()

st.subheader("Quality Distribution (Histogram)")
fig = px.histogram(df, x="quality", nbins=11, color="wine_type", barmode="overlay")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Alcohol vs Quality (Scatter)")
fig2 = px.scatter(df, x="alcohol", y="quality", color="wine_type", trendline="ols")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Feature vs Quality (Box/Violin)")
feat = st.selectbox("Choose feature", [c for c in num_cols if c != "quality"])
fig3 = px.violin(df, x="wine_type", y=feat, color="wine_type", box=True, points="all")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Correlation Heatmap (Top 10 by |corr| with quality)")
corr = df[num_cols].corr(numeric_only=True)["quality"].abs().sort_values(ascending=False).head(10).index
fig4 = px.imshow(df[corr].corr(), text_auto=True, aspect="auto", color_continuous_scale="RdBu", origin="lower")
st.plotly_chart(fig4, use_container_width=True)
