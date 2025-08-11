import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# ----------------- Page -----------------
st.title("🔮 Predict — Is this red wine **Good**?")

# Clear note: label definition vs decision threshold
st.caption(
    "Label definition: In the training data, wines with **quality ≥ 7** were labeled as **'Good'**. "
    "At prediction time, we predict 'Good' when the model's **P(Good) ≥ threshold** (default 0.50 matches)."
)

def lottie(url: str, height=180):
    try:
        from streamlit_lottie import st_lottie
        import requests
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            st_lottie(r.json(), height=height, loop=False, quality="high")
    except Exception:
        pass

# ----------------- Load model & data -----------------
MODEL_PATH = Path("model.pkl")
DATA_PATH  = Path("data/winequality-red.csv")
FEAT_PATH  = Path("feature_order.pkl")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
feature_order = joblib.load(FEAT_PATH) if FEAT_PATH.exists() else None

# columns used by the model (exclude target)
X_cols = [c for c in df.columns if c != "quality"]
if feature_order:
    X_cols = [c for c in feature_order if c in X_cols]

desc = df[X_cols].describe()

# --------- Defaults from "Good" wines only ----------
good_df    = df[df["quality"] >= 7]
good_means = good_df[X_cols].mean().to_dict()
all_means  = df[X_cols].mean()

# --------- Decision threshold (sidebar) ----------
with st.sidebar:
    st.markdown("### Decision threshold")
    threshold = st.slider(
        "Probability cutoff for 'Good'",
        min_value=0.10, max_value=0.90, value=0.50, step=0.01
    )
    st.caption(    "Prediction rule:  \n**P(Good) ≥ threshold → 'Good'**."
)

# Helper to compute P(Good) safely (Pipeline or bare estimator)
def proba_good_of(model_obj, x_df):
    clf = getattr(model_obj, "named_steps", {}).get("clf", model_obj)
    proba = clf.predict_proba(x_df)
    classes = list(clf.classes_)  # e.g., [0,1] or ['Not Good','Good']
    try:
        idx = classes.index(1) if 1 in classes else classes.index("Good")
    except ValueError:
        # Fallback: assume positive class is the last column
        idx = -1
    return float(proba[:, idx])

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
                default_val = float(good_means.get(col, desc.loc["mean", col]))

                rng = mx - mn
                step = float(max(round(rng / 100, 6), 1e-6))
                fmt  = "%.5f" if rng < 1 else "%.3f"
                help_txt = (f"Range: {mn:.5f} – {mx:.5f} | good-mean ≈ {default_val:.5f}"
                            if rng < 1 else
                            f"Range: {mn:.3f} – {mx:.3f} | good-mean ≈ {default_val:.3f}")

                default_val = min(max(default_val, mn), mx)

                inputs[col] = cols3[i % 3].number_input(
                    label=col,
                    min_value=mn, max_value=mx,
                    value=default_val,
                    step=step, format=fmt, help=help_txt
                )

    c1, c2 = st.columns([2, 2])
    with c1:
        submit = st.form_submit_button("Predict", use_container_width=True)
    with c2:
        reset = st.form_submit_button("Reset inputs", use_container_width=True)

# --------- “Good vs All” chart & table ----------
with st.expander("📊 What inputs typically make a wine *Good*? (quality ≥ 7)", expanded=False):
    comp_df = pd.DataFrame({
        "Good (≥7)": good_df[X_cols].mean(),
        "All Wines": all_means
    })
    st.dataframe(comp_df.round(4).rename_axis("Feature"), use_container_width=True)

    long_df = comp_df.reset_index().melt(id_vars="index", var_name="Group", value_name="Value")
    long_df = long_df.rename(columns={"index": "Feature"})
    fig = px.bar(long_df, x="Feature", y="Value", color="Group", barmode="group",
                 title="Average feature values: Good vs All",
                 labels={"Value": "Average"})
    fig.update_layout(height=420, xaxis_tickangle=-30, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ----------------- Predict & display -----------------
if submit:
    x_df = pd.DataFrame([inputs])[X_cols]

    # Probability of Good + thresholded label
    proba_good = proba_good_of(model, x_df)
    is_good = proba_good >= threshold
    label = "Good (quality ≥ 7)" if is_good else "Not Good (< 7)"

    # Also show model's default label (0.50 boundary)
    model_default = model.predict(x_df)[0]
    model_default_label = "Good (≥ 7)" if model_default == 1 else "Not Good (< 7)"

    left, right = st.columns([1.1, 1.4])

    with left:
        st.markdown(f"### Prediction: **{label}**")
        st.caption(f"P(Good) = {proba_good:.3f}  •  Threshold = {threshold:.2f}  •  Model default (@0.50) = {model_default_label}")
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
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_good * 100,
            number={'suffix': "%"},
            title={'text': "Probability of Good", 'font': {'size': 14}},
            domain={'x': [0, 1], 'y': [0, 0.78]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'thickness': 0.25},
                'steps': [
                    {'range': [0, 40],  'color': "rgba(239,68,68,.35)"},
                    {'range': [40, 60], 'color': "rgba(245,158,11,.30)"},
                    {'range': [60, 100],'color': "rgba(16,185,129,.30)"},
                ],
                'threshold': {'line': {'width': 3}, 'thickness': 0.75, 'value': threshold * 100}
            }
        ))
        gauge.update_layout(height=280, margin=dict(l=20, r=20, t=70, b=10))
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
    st.rerun()
