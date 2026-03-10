import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv("Student_Performance.csv")
print("Dataset shape:", df.shape)
print(df.head())

# ── 2. Preprocess ─────────────────────────────────────────────
# Encode Yes/No column
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

X = df.drop("Performance Index", axis=1)
y = df["Performance Index"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 3. Train & compare models ─────────────────────────────────
models = {
    "Linear Regression":     LinearRegression(),
    "Random Forest":         RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":     GradientBoostingRegressor(n_estimators=100, random_state=42),
}

best_model = None
best_r2 = 0
best_name = ""

print("\n── Model Comparison ──")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2  = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"{name:25s}  R² = {r2:.4f}   MAE = {mae:.2f}")
    if r2 > best_r2:
        best_r2    = r2
        best_model = model
        best_name  = name

print(f"\nBest model: {best_name}  (R² = {best_r2:.4f})")

# ── 4. Save best model ────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model saved as model.pkl")
