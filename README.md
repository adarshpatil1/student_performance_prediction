# 📚 Student Performance Predictor

A supervised Machine Learning web app that predicts a student's **Performance Index** (0–100) based on study habits and background data.

Built with Python, scikit-learn, and Streamlit.

---

## 🎯 Project Goal

Help educators identify students who may need support by predicting their performance score from simple inputs like study hours, sleep, and previous scores.

---

## 🚀 Live Demo

Run locally using the steps below.

---

## 📊 Dataset

- **Source:** [Kaggle — Student Performance (Multiple Linear Regression)](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)
- **Size:** 10,000 rows × 6 columns

| Column | Description |
|--------|-------------|
| Hours Studied | Hours studied per day (1–9) |
| Previous Scores | Score in previous exams (40–99) |
| Extracurricular Activities | Yes / No |
| Sleep Hours | Average sleep hours (1–9) |
| Sample Question Papers Practiced | Practice papers done (0–9) |
| Performance Index | Target variable — student score (0–100) |

---

## 🤖 Models Compared

| Model | R² Score | MAE |
|-------|----------|-----|
| **Linear Regression** | **0.9890** | **1.61** ✓ |
| Gradient Boosting | 0.9882 | 1.66 |
| Random Forest | 0.9861 | 1.81 |

**Best model: Linear Regression** — R² of 0.989 means the model explains 98.9% of variance in student performance.

---

## 🗂️