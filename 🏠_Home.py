import streamlit as st
import pandas as pd
from pathlib import Path

# ---------- Page setup ----------
st.set_page_config(page_title="Wine Quality (Red) – ML App", page_icon="🍷", layout="wide")

DATA_PATH = Path("data/winequality-red.csv")
df = pd.read_csv(DATA_PATH)

# ---------- (Optional) Lottie animation loader ----------
def lottie(url: str):
    try:
        from streamlit_lottie import st_lottie
        import requests
        r = requests.get(url)
        if r.status_code == 200:
            st_lottie(r.json(), height=240, loop=True, quality="high")
        else:
            st.image("https://media.giphy.com/media/kGCuRgmbnO3a0/giphy.gif")  # tasteful fallback
    except Exception:
        # streamlit-lottie not installed or offline; ignore
        pass

# ---------- Custom CSS (subtle animations + styling) ----------
st.markdown("""
<style>
/* remove default top padding for a tighter hero */
.block-container { padding-top: 1.5rem; }

/* hero gradient */
.hero {
  padding: 1.25rem 1.5rem;
  border-radius: 1.25rem;
  background: linear-gradient(135deg, rgba(99,102,241,.08), rgba(236,72,153,.08));
  border: 1px solid rgba(200,200,200,.12);
}

/* animated wine icon */
.wine {
  display:inline-block;
  animation: floaty 3s ease-in-out infinite;
}
@keyframes floaty {
  0%   { transform: translateY(0px) rotate(0deg); }
  50%  { transform: translateY(-4px) rotate(1deg); }
  100% { transform: translateY(0px) rotate(0deg); }
}

/* pill badges */
.badge {
  display:inline-block;
  padding: .35rem .6rem;
  margin: .15rem .35rem .15rem 0;
  border-radius: 999px;
  font-size: .82rem;
  border: 1px solid rgba(200,200,200,.18);
  background: rgba(255,255,255,.04);
}

/* glass card */
.card {
  padding: 1rem 1.1rem;
  border-radius: 1rem;
  border: 1px solid rgba(200,200,200,.12);
  background: rgba(255,255,255,.03);
}

/* fade-in */
.fade-in { animation: fade 400ms ease 1; }
@keyframes fade { from {opacity: .0; transform: translateY(4px);} to {opacity:1;} }
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown(
    f"""
<div class="hero fade-in">
  <h1 style="margin:0 0 .3rem 0; font-size: 2.2rem;">
    <span class="wine">🍷</span> Wine Quality (Red) — Machine Learning App
  </h1>
  <p style="margin:.25rem 0 0 0; opacity:.9;">
    Predict whether a red wine is <b>Good</b> (quality ≥ 7) or <b>Not Good</b> from its chemical properties.
  </p>
</div>
""",
    unsafe_allow_html=True
)

st.write("")  # tiny spacer

# ---------- Top row: Animation + Quick CTAs ----------
left, right = st.columns([1.1, 1.4])

with left:
    st.markdown("#### ")
    # 👉 Replace the URL with any Lottie wine/grapes animation you like
    lottie("https://assets1.lottiefiles.com/packages/lf20_ia8dck.json")  # optional; silently ignored if not available

with right:
    st.markdown("#### Quick start")
    c1, c2, c3, c4 = st.columns(4)
    # Modern page links (Streamlit 1.30+). If on older version, use st.page_link or st.write with markdown URLs.
    try:
        st.page_link("pages/1_📊_Data_Explorer.py", label="📊 Data Explorer")
        st.page_link("pages/2_📈_Visualizations.py", label="📈 Visualizations")
        st.page_link("pages/3_🔮_Predict.py", label="🔮 Predict")
        st.page_link("pages/4_📎_Model_Performance.py", label="📎 Model Performance")
    except Exception:
        # Fallback: show four small buttons that navigate by instructions
        c1.button("📊 Data Explorer")
        c2.button("📈 Visualizations")
        c3.button("🔮 Predict")
        c4.button("📎 Performance")
    st.write("")
    st.markdown(
        """
<div class="card">
  <span class="badge">scikit‑learn</span>
  <span class="badge">Random Forest</span>
  <span class="badge">StandardScaler</span>
  <span class="badge">Plotly</span>
  <span class="badge">Streamlit</span>
</div>
""",
        unsafe_allow_html=True
    )

# ---------- Metrics ----------
m1, m2, m3 = st.columns(3)
m1.metric("Rows", f"{df.shape[0]}")
m2.metric("Columns", f"{df.shape[1]}")
m3.metric("Target", "Binary: quality ≥ 7")

# ---------- How it works ----------
st.markdown("### How it works")
st.markdown(
    """
1. **Explore** the data and correlations to understand what drives quality (e.g., alcohol, sulphates, volatile acidity).  
2. **Predict** with your own inputs — we score probability of being **Good**.  
3. **Review** performance (Accuracy, F1, ROC‑AUC), confusion matrix, and ROC curve.  
    """.strip()
)

# ---------- Dataset preview ----------
with st.expander("🔎 Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)

# ---------- Footer ----------
st.caption("Built for your ML deployment assignment — Streamlit • scikit‑learn • Plotly")
