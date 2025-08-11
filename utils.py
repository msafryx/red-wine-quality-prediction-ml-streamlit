from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_DIR = Path("data")
RED = DATA_DIR / "winequality-red.csv"
WHITE = DATA_DIR / "winequality-white.csv"
MODEL_PATH = Path("models/model.pkl")
META_PATH = Path("models/meta.pkl")  # store task, features, etc.

@st.cache_data(show_spinner=False)
def load_wine() -> pd.DataFrame:
    red = pd.read_csv(RED, sep=';')
    white = pd.read_csv(WHITE, sep=';')
    red["wine_type"] = "red"
    white["wine_type"] = "white"
    df = pd.concat([red, white], ignore_index=True)
    # Make column names friendly
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df

def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )

def make_classification_target(df: pd.DataFrame, low_max=5, high_min=7, col="quality") -> pd.Series:
    """Bin quality into 3 classes: low (≤5), medium (6), high (≥7)."""
    y = pd.cut(
        df[col],
        bins=[-np.inf, low_max, high_min-1e-9, np.inf],
        labels=["low", "medium", "high"]
    )
    return y.astype("category")

def save_artifacts(model, meta: dict):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(meta, META_PATH)

def load_artifacts():
    if MODEL_PATH.exists() and META_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(META_PATH)
    return None, None
