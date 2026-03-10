import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

# ── Train and save model if not present ──────────────────────
def train_model():
    df = pd.read_csv("Student_Performance.csv")
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    X = df.drop("Performance Index", axis=1)
    y = df["Performance Index"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model

# ── Load or train model ───────────────────────────────────────
if os.path.exists("model.pkl"):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
else:
    model = train_model()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Student Performance Predictor", page_icon="📚", layout="centered")

st.title("Student Performance Predictor")
st.markdown("Enter student details below to predict their **Performance Index** (0 – 100).")
st.markdown("---")

# ── Input form ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    hours_studied   = st.slider("Hours Studied per Day", 1, 9, 5)
    sleep_hours     = st.slider("Sleep Hours per Day", 1, 9, 7)
    extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

with col2:
    previous_scores = st.slider("Previous Scores", 40, 99, 70)
    sample_papers   = st.slider("Sample Papers Practiced", 0, 9, 3)

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────
extra = 1 if extracurricular == "Yes" else 0
features = np.array([[hours_studied, previous_scores, extra, sleep_hours, sample_papers]])

if st.button("Predict Performance", use_container_width=True):
    prediction = round(float(model.predict(features)[0]), 1)

    st.markdown("### Result")

    if prediction >= 75:
        st.success(f"Predicted Performance Index: **{prediction} / 100**")
    elif prediction >= 50:
        st.warning(f"Predicted Performance Index: **{prediction} / 100**")
    else:
        st.error(f"Predicted Performance Index: **{prediction} / 100**")

    low  = max(0,   round(prediction - 3, 1))
    high = min(100, round(prediction + 3, 1))
    st.caption(f"Estimated range: {low} – {high}")

st.markdown("---")
st.caption("Built with Python · scikit-learn · Streamlit")
