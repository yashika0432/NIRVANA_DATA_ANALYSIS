"""
predict.py
──────────
Load trained models and run predictions.

Usage:
  python predict.py

The predict_attack() function can also be imported directly:
  from predict import predict_attack
  result = predict_attack({"sleep_hours": 4, "stress_level": 9, ...})
"""

import joblib
import numpy as np


# ── Load models ───────────────────────────────────────────────

model         = joblib.load("model.pkl")           # days regressor
trigger_model = joblib.load("trigger_model.pkl")   # trigger classifier
scaler        = joblib.load("scaler.pkl")           # feature scaler
encoder       = joblib.load("label_encoder.pkl")   # trigger label encoder


# ── Risk label helper ─────────────────────────────────────────

def risk_label(score: float) -> str:
    if score >= 7.5:
        return "High"
    elif score >= 5.0:
        return "Medium"
    return "Low"


# ── Main prediction function ──────────────────────────────────

def predict_attack(data: dict) -> dict:
    """
    Predict the next anxiety attack profile from behavioral data.

    Args:
        data: dict with keys:
            sleep_hours      (float, 0–24)
            stress_level     (float, 0–10)
            anxiety_level    (float, 0–10)
            caffeine_intake  (float, cups/day)
            mood_score       (float, 0–10)
            trigger_encoded  (int, from LabelEncoder)

    Returns:
        dict with:
            expected_days     — days until next attack (float)
            expected_trigger  — trigger type name (str)
            risk_score        — composite 0–10 score (float)
            risk_label        — "Low" / "Medium" / "High" (str)
    """

    # Build feature array — must match training order exactly
    features = np.array([[
        data["sleep_hours"],
        data["stress_level"],
        data["anxiety_level"],
        data["caffeine_intake"],
        data["mood_score"],
        data["trigger_encoded"],
    ]])

    # Scale using the saved scaler
    scaled = scaler.transform(features)

    # Model 1: days until next attack
    days_prediction = model.predict(scaled)[0]

    # Model 2: most likely trigger type
    trigger_prediction = trigger_model.predict(scaled)[0]
    trigger_name = encoder.inverse_transform([int(trigger_prediction)])[0]

    # Risk score (weighted formula — same used in EDA)
    score = (
        data["stress_level"] * 0.4
        + data["anxiety_level"] * 0.4
        + (10 - data["sleep_hours"]) * 0.2
    )
    score = round(min(score, 10.0), 1)

    return {
        "expected_days":    round(float(days_prediction), 1),
        "expected_trigger": trigger_name,
        "risk_score":       score,
        "risk_label":       risk_label(score),
    }


# ── Demo predictions ──────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  NIRVANA — PREDICTION DEMO")
    print("=" * 55)

    scenarios = [
        {
            "label": "High risk — sleep-deprived, high stress",
            "data": {
                "sleep_hours": 4,
                "stress_level": 9,
                "anxiety_level": 8,
                "caffeine_intake": 5,
                "mood_score": 3,
                "trigger_encoded": 2,
            },
        },
        {
            "label": "Moderate risk — moderate load",
            "data": {
                "sleep_hours": 5.5,
                "stress_level": 6,
                "anxiety_level": 5,
                "caffeine_intake": 3,
                "mood_score": 5,
                "trigger_encoded": 1,
            },
        },
        {
            "label": "Low risk — healthy baseline",
            "data": {
                "sleep_hours": 7.5,
                "stress_level": 3,
                "anxiety_level": 2,
                "caffeine_intake": 1,
                "mood_score": 8,
                "trigger_encoded": 0,
            },
        },
    ]

    for s in scenarios:
        result = predict_attack(s["data"])
        print(f"\n  {s['label']}")
        print(f"  Input  : sleep={s['data']['sleep_hours']}h | "
              f"stress={s['data']['stress_level']} | "
              f"anxiety={s['data']['anxiety_level']}")
        print(f"  Output :")
        print(f"    Risk score      : {result['risk_score']} / 10  ({result['risk_label']})")
        print(f"    Expected trigger: {result['expected_trigger']}")
        print(f"    Next attack in  : {result['expected_days']} days")

    print("\n" + "─" * 55)
    print("  Why this prediction?")
    print("─" * 55)
    sample = scenarios[0]["data"]
    reasons = []
    if sample["sleep_hours"] < 5:
        reasons.append(f"  • Sleep is critically low ({sample['sleep_hours']}h) — "
                       f"severe sleep deprivation directly elevates cortisol and anxiety.")
    if sample["stress_level"] >= 7:
        reasons.append(f"  • Stress is elevated ({sample['stress_level']}/10) — "
                       f"sustained high stress compounds with poor sleep.")
    if sample["caffeine_intake"] >= 5:
        reasons.append(f"  • Caffeine is high ({sample['caffeine_intake']} cups) — "
                       f"amplifies anxiety symptoms and disrupts sleep quality.")
    if sample["anxiety_level"] >= 7:
        reasons.append(f"  • Anxiety level ({sample['anxiety_level']}/10) is already elevated, "
                       f"indicating an ongoing episode or high baseline.")
    for r in reasons:
        print(r)
    print()