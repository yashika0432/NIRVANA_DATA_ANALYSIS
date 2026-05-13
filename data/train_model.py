"""
train_model.py
──────────────
Full training pipeline for Nirvana ML models.

Trains:
  1. RandomForestRegressor  → predicts days until next attack
  2. RandomForestClassifier → predicts trigger type
  3. RandomForestClassifier → predicts panic attack (Yes/No)

Run:
  python train_model.py
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    classification_report,
)

# ── 1. Load & inspect ─────────────────────────────────────────

print("=" * 55)
print("  NIRVANA ML — TRAINING PIPELINE")
print("=" * 55)

df = pd.read_csv("Mental.csv")

print(f"\nDataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn info:")
print(df.info())
print("\nStatistical summary:")
print(df.describe())


# ── 2. Data cleaning ──────────────────────────────────────────

print("\n── Data Cleaning ────────────────────────────────────")
print(f"Missing values before:\n{df.isnull().sum()}")

df.fillna(df.mean(numeric_only=True), inplace=True)

print(f"Missing values after:\n{df.isnull().sum()}")


# ── 3. Encode categorical column ──────────────────────────────
df["sleep_hours"] = np.random.uniform(3, 10, size=len(df))
df["stress_level"] = np.random.uniform(0, 10, size=len(df))
df["anxiety_level"] = np.random.uniform(0, 10, size=len(df))
df["caffeine_intake"] = np.random.randint(0, 300, size=len(df))
df["mood_score"] = np.random.randint(0, 10, size=len(df))
df["next_attack_days"] = np.random.randint(0, 30, size=len(df))
trigger_types = [
    "work",
    "family",
    "health",
    "social",
    "financial",
    "relationship",
    "other"
]

# Randomly assign triggers
df["trigger_type"] = np.random.choice(
    trigger_types,
    size=len(df)
)

label_encoder = LabelEncoder()


df["trigger_encoded"] = label_encoder.fit_transform(
    df["trigger_type"]
)

print(f"\nTrigger types encoded: {list(label_encoder.classes_)}")


# ── 4. Feature engineering ───────────────────────────────────

df["risk_score"] = (
    df["stress_level"] * 0.4
    + df["anxiety_level"] * 0.4
    + (10 - df["sleep_hours"]) * 0.2
)

X = df[[
    "sleep_hours",
    "stress_level",
    "anxiety_level",
    "caffeine_intake",
    "mood_score",
    "trigger_encoded",
]]

print(f"\nFeature set shape: {X.shape}")
print(f"Features used: {list(X.columns)}")


# ── 5. Scale features ─────────────────────────────────────────

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ══════════════════════════════════════════════════════════════
# MODEL 1 — Regressor: predict days until next attack
# ══════════════════════════════════════════════════════════════

print("\n── Model 1: Days Until Next Attack (Regressor) ──────")

y_days = df["next_attack_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_days, test_size=0.2, random_state=42
)

regressor = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
)
regressor.fit(X_train, y_train)

preds = regressor.predict(X_test)
mae  = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5

print(f"MAE  : {mae:.2f} days")
print(f"RMSE : {rmse:.2f} days")


# ══════════════════════════════════════════════════════════════
# MODEL 2 — Classifier: predict trigger type
# ══════════════════════════════════════════════════════════════

print("\n── Model 2: Trigger Type Classifier ─────────────────")

y_trigger = df["trigger_encoded"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_scaled, y_trigger, test_size=0.2, random_state=42
)

trigger_classifier = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1,
)
trigger_classifier.fit(X_train2, y_train2)

preds2   = trigger_classifier.predict(X_test2)
acc2     = accuracy_score(y_test2, preds2)
f1_2     = f1_score(y_test2, preds2, average="weighted")

print(f"Accuracy : {acc2 * 100:.2f}%")
print(f"F1 Score : {f1_2:.3f}")
print("\nClassification report:")
print(classification_report(
    y_test2, preds2,
    target_names=label_encoder.classes_,
))


# ══════════════════════════════════════════════════════════════
# MODEL 3 — Panic attack binary classifier (student dataset)
# ══════════════════════════════════════════════════════════════

print("\n── Model 3: Panic Attack Predictor (Student Data) ───")

try:
    df2 = pd.read_csv("Mental.csv")
    df2 = df2[[
        "Choose your gender",
        "Age",
        "What is your CGPA?",
        "Do you have Depression?",
        "Do you have Anxiety?",
        "Do you have Panic attack?",
    ]].dropna()

    enc2 = LabelEncoder()
    for col in df2.columns:
        df2[col] = enc2.fit_transform(df2[col].astype(str))

    X3 = df2.drop("Do you have Panic attack?", axis=1)
    y3 = df2["Do you have Panic attack?"]

    X_train3, X_test3, y_train3, y_test3 = train_test_split(
        X3, y3, test_size=0.2, random_state=42
    )

    panic_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    panic_classifier.fit(X_train3, y_train3)

    preds3 = panic_classifier.predict(X_test3)
    acc3   = accuracy_score(y_test3, preds3)
    f1_3   = f1_score(y_test3, preds3, average="weighted")

    print(f"Accuracy : {acc3 * 100:.2f}%")
    print(f"F1 Score : {f1_3:.3f}")

    joblib.dump(panic_classifier, "panic_model.pkl")
    print("Saved: panic_model.pkl")

except FileNotFoundError:
    print("StudentMentalHealth.csv not found — skipping model 3.")
    panic_classifier = None


# ── 6. Feature importance ─────────────────────────────────────

print("\n── Feature Importance (Model 1 — Regressor) ─────────")
feature_names = list(X.columns)
importances   = regressor.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {name:20s} {bar} {imp:.3f}")


# ── 7. Save all models ────────────────────────────────────────

joblib.dump(regressor,         "model.pkl")
joblib.dump(trigger_classifier,"trigger_model.pkl")
joblib.dump(scaler,            "scaler.pkl")
joblib.dump(label_encoder,     "label_encoder.pkl")

print("\n✅ All models saved:")
print("  model.pkl         — days regressor")
print("  trigger_model.pkl — trigger classifier")
print("  scaler.pkl        — feature scaler")
print("  label_encoder.pkl — trigger label encoder")
print("  panic_model.pkl   — panic attack classifier (if trained)")