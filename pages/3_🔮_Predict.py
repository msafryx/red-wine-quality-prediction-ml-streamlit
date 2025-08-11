import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go

# ----------------- Page -----------------
st.title("🔮 Predict — Is this red wine **Good**?")
st.caption("Binary target: quality ≥ 7 → Good")

def lottie(url: str, height=180):
    try:
        from streamlit_lottie import st_lottie
        import requests
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            st_lottie(r.json(), height=height, loop=False, quality="high")
    except Exception:
        pass

# ----------------- Load model & data stats -----------------
MODEL_PATH = Path("model.pkl")
DATA_PATH  = Path("data/winequality-red.csv")
FEAT_PATH  = Path("feature_order.pkl")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
feature_order = joblib.load(FEAT_PATH) if FEAT_PATH.exists() else None

X_cols = [c for c in df.columns if c != "quality"]
if feature_order:
    X_cols = [c for c in feature_order if c in X_cols]

desc = df[X_cols].describe()

# ----------------- UI: grouped inputs -----------------
st.markdown("#### Enter chemical properties")

groups = {
    "Acidity": ["fixed acidity", "volatile acidity", "citric acid", "pH"],
    "Sugar & Chlorides": ["residual sugar", "chlorides"],
    "Sulfur Dioxide": ["free sulfur dioxide", "total sulfur dioxide"],
    "Body & Structure": ["density", "sulphates", "alcohol"],
}
groups = {g: [c for c in cols if c in X_cols] for g, cols in groups.items()}

inputs = {}
with st.form("predict_form", clear_on_submit=False):
    tabs = st.tabs(list(groups.keys()))
    for ti, (gname, cols) in enumerate(groups.items()):
        with tabs[ti]:
            cols3 = st.columns(3)
            for i, col in enumerate(cols):
                mn = float(desc.loc["min", col])
                mx = float(desc.loc["max", col])
                mean = float(desc.loc["mean", col])
                rng = mx - mn
                step = float(max(round(rng / 100, 6), 1e-6))
                fmt  = "%.5f" if rng < 1 else "%.3f"
                help_txt = (f"Range: {mn:.5f} – {mx:.5f} | mean ≈ {mean:.5f}"
                            if rng < 1 else
                            f"Range: {mn:.3f} – {mx:.3f} | mean ≈ {mean:.3f}")
                inputs[col] = cols3[i % 3].number_input(
                    label=col, min_value=mn, max_value=mx,
                    value=mean, step=step, format=fmt, help=help_txt
                )

    c1, c2 = st.columns([2, 2])
    with c1:
        submit = st.form_submit_button("Predict", use_container_width=True)
    with c2:
        reset = st.form_submit_button("Reset inputs", use_container_width=True)

# ----------------- Predict & display -----------------
if submit:
    x_df = pd.DataFrame([inputs])[X_cols]
    predicted_class = model.predict(x_df)[0]
    proba_good = float(model.predict_proba(x_df)[:, 1])
    is_good = bool(predicted_class == 1)
    label = "Good (quality ≥ 7)" if is_good else "Not Good (< 7)"

    left, right = st.columns([1.1, 1.4])

    with left:
        st.markdown(f"### Prediction: **{label}**")
        st.caption(f"Model probability: {proba_good:.3f}")
        if is_good:
            lottie("https://assets2.lottiefiles.com/packages/lf20_3vbOcw.json", height=140)
            st.balloons()
        else:
            try:
                from streamlit_lottie import st_lottie
                import json, os
                if os.path.exists("assets/bad.json"):
                    with open("assets/bad.json","r",encoding="utf-8") as f:
                        st_lottie(json.load(f), height=160, loop=False, quality="high")
                else:
                    lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json", height=160)
            except Exception:
                pass
            st.markdown("""
            <style>
              .shake {display:inline-block; animation: shake 0.6s ease-in-out 0s 1;}
              @keyframes shake {
                10%, 90% { transform: translateX(-1px); }
                20%, 80% { transform: translateX(2px); }
                30%, 50%, 70% { transform: translateX(-4px); }
                40%, 60% { transform: translateX(4px); }
              }
            </style>
            <div class="shake" style="font-size:2rem;">❌</div>
            """, unsafe_allow_html=True)

    with right:
        # Move gauge LOWER via domain.y, and give more top margin; also shrink title font.
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_good * 100,
            number={'suffix': "%"},
            title={'text': "Probability of Good", 'font': {'size': 14}},
            domain={'x': [0, 1], 'y': [0, 0.78]},  # <- shifts the meter down
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'thickness': 0.25},
                'steps': [
                    {'range': [0, 40],  'color': "rgba(239,68,68,.35)"},
                    {'range': [40, 60], 'color': "rgba(245,158,11,.30)"},
                    {'range': [60, 100],'color': "rgba(16,185,129,.30)"},
                ],
                'threshold': {'line': {'width': 3}, 'thickness': 0.75, 'value': proba_good * 100}
            }
        ))
        gauge.update_layout(
            height=280,                      # a bit taller
            margin=dict(l=20, r=20, t=70, b=10)  # more top space so nothing clips
        )
        st.plotly_chart(gauge, use_container_width=True)

    # Feature importances (if available)
    try:
        clf = model.named_steps.get("clf", None)
        if clf is not None and hasattr(clf, "feature_importances_"):
            importances = pd.Series(clf.feature_importances_, index=X_cols)\
                            .sort_values(ascending=False).head(10)
            st.markdown("#### Top feature importances")
            st.bar_chart(importances)
    except Exception:
        pass

    with st.expander("Show input vector"):
        st.dataframe(x_df.T.rename(columns={0: "value"}))

elif reset:
    st.rerun()  # modern Streamlit reset
