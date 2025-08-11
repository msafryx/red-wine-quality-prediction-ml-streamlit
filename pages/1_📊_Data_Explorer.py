import streamlit as st
from utils import load_wine
import pandas as pd

st.header("📊 Data Explorer")

df = load_wine()

c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(df))
c2.metric("Columns", df.shape[1])
c3.write("dtypes"); c3.write(df.dtypes)

st.subheader("Sample")
st.dataframe(df.head(), use_container_width=True)

st.subheader("Interactive Filter")
cols = st.multiselect("Columns", df.columns.tolist(), default=list(df.columns)[:8])
query = st.text_input("Row filter (pandas query)", value="")
try:
    filtered = df.query(query) if query else df
    st.dataframe(filtered[cols] if cols else filtered, use_container_width=True)
except Exception as e:
    st.warning(f"Filter error: {e}")
