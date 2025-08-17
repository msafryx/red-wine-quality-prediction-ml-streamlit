import json
import streamlit as st
import pandas as pd
from pathlib import Path

# -------------------- Page setup --------------------
st.set_page_config(
    page_title=" Red Wine Quality Prediction  – ML App",
    page_icon="🍷",
    layout="wide"
)

DATA_PATH = Path("data/winequality-red.csv")
df = pd.read_csv(DATA_PATH)

# -------------------- Lottie loader (local first, URL fallback) --------------------
def load_lottie_local(path: Path):
    """Return parsed Lottie JSON from a local file, or None if not available."""
    try:
        from streamlit_lottie import st_lottie  # ensure installed
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def render_lottie(obj_or_url, *, height=260):
    """Render a Lottie object (dict) OR download from URL if a string is given."""
    try:
        from streamlit_lottie import st_lottie
        if isinstance(obj_or_url, dict):
            st_lottie(obj_or_url, height=height, loop=True, quality="high")
        elif isinstance(obj_or_url, str):
            import requests
            r = requests.get(obj_or_url, timeout=6)
            if r.ok:
                st_lottie(r.json(), height=height, loop=True, quality="high")
    except Exception:
        # If streamlit-lottie isn't installed or anything fails, we just skip silently
        pass

# Path to your local Lottie JSON (keep the spaces if your file is named that way)
LOTTIE_PATH = Path("assets/wine-animation.json")
lottie_local = load_lottie_local(LOTTIE_PATH)
# Optional backup URL in case the local file isn't present
LOTTIE_FALLBACK_URL = "https://assets10.lottiefiles.com/packages/lf20_bqmgf5tx.json"

# -------------------- Global CSS --------------------
st.markdown("""
<style>
/* tighten page */
.block-container { padding-top: 1.2rem; }

/* center hero */
.hero {
  position: relative;
  padding: 2.0rem 2rem 1.5rem 2rem;
  border-radius: 18px;
  text-align: center;
  background: radial-gradient(1100px 400px at 50% -20%, rgba(236,72,153,.10), rgba(79,70,229,.10) 45%, transparent 70%),
              linear-gradient(135deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.12);
  box-shadow: 0 10px 30px rgba(0,0,0,.15);
}

/* title + subtitle */
.hero h1 {
  font-size: 2.4rem;
  margin: 0 0 .3rem 0;
  letter-spacing: .2px;
}
.hero p {
  margin: .25rem 0 0 0;
  opacity: .90;
}

/* big emoji animation */
.bounce {
  display:inline-block;
  animation: bounce 2.6s ease-in-out infinite;
}
@keyframes bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-8px); }
}

/* metric cards */
.kpi {
  padding: 1rem 1.1rem;
  background: rgba(255,255,255,.03);
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 14px;
}

/* nav buttons */
.navgrid {
  display: grid;
  grid-template-columns: repeat(4, minmax(160px, 1fr));
  gap: .75rem;
  margin-top: .5rem;
}
.navbtn {
  display: inline-flex; align-items:center; justify-content:center;
  gap:.5rem;
  padding:.85rem 1rem;
  border-radius: 12px;
  border:1px solid rgba(255,255,255,.12);
  background: rgba(255,255,255,.04);
  text-decoration:none !important;
  transition: transform .15s ease, background .15s ease, border-color .15s ease;
}
.navbtn:hover { transform: translateY(-2px); background: rgba(255,255,255,.07); border-color: rgba(255,255,255,.18); }

/* tech badges */
.badges { margin-top:.5rem; }
.badge {
  display:inline-block;
  padding:.35rem .6rem;
  margin:.18rem .25rem 0 .25rem;
  border-radius:999px;
  font-size:.82rem;
  border:1px solid rgba(255,255,255,.16);
  background: rgba(255,255,255,.04);
}

/* fade-in on load */
.fade-in { animation: fade .45s ease 1; }
@keyframes fade { from {opacity:0; transform: translateY(6px);} to {opacity:1; transform:none;} }
</style>
""", unsafe_allow_html=True)

# -------------------- HERO --------------------
st.markdown(
    """
<div class="hero fade-in">
  <h1><span class="bounce">🍷</span> Red Wine Quality Prediction — Machine Learning App</h1>
  <p>Predict whether a red wine is <b>Good</b> (quality ≥ 7) or <b>Not Good</b> from its chemical properties.</p>
</div>
""",
    unsafe_allow_html=True
)

st.write("")  # small spacer

# -------------------- Animation + Quick links --------------------
col_anim, col_nav = st.columns([1, 1.2], vertical_alignment="center")

with col_anim:
    st.subheader(" ")
    # Prefer local Lottie; if missing, use fallback URL
    render_lottie(lottie_local if lottie_local else LOTTIE_FALLBACK_URL, height=240)

with col_nav:
    st.subheader("Quick Start")
    # Use page_link where available; if not, show inert buttons
    try:
        st.markdown('<div class="navgrid">', unsafe_allow_html=True)
        st.page_link("pages/1_📊_Data_Explorer.py", label="📊 Data Explorer", help="Browse & filter the dataset")
        st.page_link("pages/2_📈_Visualizations.py", label="📈 Visualizations", help="See charts & correlations")
        st.page_link("pages/3_🔮_Predict.py", label="🔮 Predict", help="Enter features and predict quality")
        st.page_link("pages/4_📎_Model_Performance.py", label="📎 Model Performance", help="Metrics, ROC/PR curves")
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception:
        st.markdown(
            """
            <div class="navgrid">
              <div class="navbtn">📊 Data Explorer</div>
              <div class="navbtn">📈 Visualizations</div>
              <div class="navbtn">🔮 Predict</div>
              <div class="navbtn">📎 Performance</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="badges">
          <span class="badge">scikit-learn</span>
          <span class="badge">Random Forest</span>
          <span class="badge">StandardScaler</span>
          <span class="badge">Plotly</span>
          <span class="badge">Streamlit</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")  # spacer

# -------------------- KPIs --------------------
k1, k2, k3 = st.columns(3)
with k1:
    st.container(height=1, border=False)
    st.metric("Rows", f"{df.shape[0]}")
with k2:
    st.metric("Columns", f"{df.shape[1]}")
with k3:
    st.metric("Target Definition", "Good if quality ≥ 7")

st.write("")

# -------------------- About / How it works --------------------
st.subheader("What is This App?")
st.markdown(
    """
This web app uses a trained machine-learning model on the **Wine Quality (Red)** dataset to predict whether a wine is
**Good (quality ≥ 7)** based on its chemical properties.

**Use the sidebar or Quick Start**:
- **Data Explorer** — browse and filter the dataset  
- **Visualizations** — understand feature distributions & correlations  
- **Predict** — try your own feature values and get a probability of being Good  
- **Model Performance** — see accuracy, F1, ROC-AUC, confusion matrix, ROC & PR curves
"""
)

# -------------------- Dataset preview (expanded by default) --------------------
with st.expander("🔎 Preview dataset", expanded=True):
    st.dataframe(df.head(25), use_container_width=True)

# -------------------- Footer --------------------
st.caption("Built with ❤️ By Muhammed Safry")
