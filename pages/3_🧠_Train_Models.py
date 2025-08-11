import streamlit as st
import pandas as pd
from utils import load_wine, build_preprocess, make_classification_target, save_artifacts
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.header("🧠 Train Models")

df = load_wine()

task = st.radio("Task", ["Classification (low/medium/high)", "Regression (numeric quality)"])
target_col = "quality"

# Features: everything except target
X = df.drop(columns=[target_col])
if task.startswith("Classification"):
    y = make_classification_target(df, col=target_col)
else:
    y = df[target_col]

test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.number_input("Random state", 0, 9999, 42, 1)

models = {}
if task.startswith("Classification"):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=250, random_state=42)
        # (optional) add XGBoostClassifier
    }
else:
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=42)
        # (optional) add XGBRegressor
    }

if st.button("Train & Compare"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if task.startswith("Classification") else None
    )

    preprocess = build_preprocess(X_train)

    results = []
    best_name, best_model, best_score = None, None, -1e9

    for name, est in models.items():
        pipe = Pipeline([("prep", preprocess), ("model", est)])
        if task.startswith("Classification"):
            cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results.append({"Model": name, "CV_Accuracy": cv.mean(), "Holdout_Accuracy": acc})
            score = acc
        else:
            cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            rmse = mean_squared_error(y_test, preds, squared=False)
            results.append({"Model": name, "CV_RMSE": -cv.mean(), "Holdout_RMSE": rmse})
            score = -rmse

        if score > best_score:
            best_name, best_model, best_score = name, pipe, score

    st.success("Training finished.")
    st.dataframe(pd.DataFrame(results).sort_values(list(results[0].keys())[-1], ascending=task.startswith("Regression")), use_container_width=True)

    st.session_state["best_model"] = best_model
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["task"] = "classification" if task.startswith("Classification") else "regression"

    if st.button("Save Best Model"):
        meta = {
            "task": st.session_state["task"],
            "feature_columns": X.columns.tolist(),
            "target": target_col
        }
        save_artifacts(best_model, meta)
        st.toast(f"Saved best {best_name} to models/model.pkl", icon="✅")
