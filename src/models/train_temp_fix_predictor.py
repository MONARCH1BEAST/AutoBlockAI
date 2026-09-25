"""
src/models/train_temp_fix_predictor.py

Trains XGBoost to predict whether a temporary fix will fail within
the next 7 days.

Design:
  Rather than predicting the artificially-generated lifespan column
  (which would be circular), we simulate a realistic failure process
  driven by causal factors (traffic, days since fix, severity, etc.),
  then train the model to learn that process.

Target: failed_within_7_days (binary)
Outputs:
  - src/models/saved/temp_fix_model.pkl
  - src/models/saved/temp_fix_features.pkl
  - output/reports/temp_fix_model_performance.txt
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

np.random.seed(42)

print("=" * 70)
print("STAGE 2: TRAINING TEMP-FIX FAILURE PREDICTOR")
print("=" * 70)

# ============================================================
# 1. LOAD ENRICHED TASKS
# ============================================================
print("\n[1/6] Loading enriched tasks...")
tasks = pd.read_csv("data/processed/maintenance_tasks_enriched.csv")

# Filter to only tasks with temp fixes
df = tasks[tasks["temporary_fix_applied"] == True].copy()
print(f"  Total tasks with temp fixes: {len(df)}")

# ============================================================
# 2. SIMULATE REALISTIC FAILURE PROCESS
# ============================================================
print("\n[2/6] Simulating temp-fix failure process...")

# Failure probability from real-world factors
def compute_failure_probability(row):
    """
    Compute probability that this temp fix fails in the next 7 days.
    
    Factors:
      - temp_fix_age_ratio: how far through its lifespan (dominant)
      - corridor_criticality: busier corridor = more wear
      - severity: worse original defect = weaker temp fix
      - dept_factor: signals fail faster than civil
    """
    ratio = row["temp_fix_age_ratio"]

    # Base probability from age ratio
    if ratio < 0.3:
        age_prob = 0.02
    elif ratio < 0.6:
        age_prob = 0.10
    elif ratio < 0.8:
        age_prob = 0.30
    elif ratio < 0.95:
        age_prob = 0.65
    else:
        age_prob = 0.92

    # Corridor traffic multiplier (busy corridors accelerate failure)
    traffic_mult = 1.0 + 0.5 * row["criticality"]

    # Severity multiplier (worse defect = weaker fix)
    severity_mult = 0.8 + 0.15 * row["severity"]

    # Department multiplier
    dept_mult = {
        "Engineering": 1.0,
        "Signal & Telecom": 1.3,        # signals fail faster
        "Traction Distribution": 1.15,
    }.get(row["department"], 1.0)

    # Combined probability (cap at 0.99)
    p = min(0.99, age_prob * traffic_mult * severity_mult * dept_mult)

    return p

df["failure_prob"] = df.apply(compute_failure_probability, axis=1)

# Simulate the actual failure event
df["failed_within_7_days"] = (np.random.random(len(df)) < df["failure_prob"]).astype(int)

print(f"  Simulated failures in next 7 days: {df['failed_within_7_days'].sum()} ({df['failed_within_7_days'].mean()*100:.1f}%)")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n[3/6] Engineering features...")

DEPARTMENTS = ["Engineering", "Signal & Telecom", "Traction Distribution"]

def build_features(row):
    features = {
        "severity": row["severity"],
        "days_overdue": min(row["days_overdue"], 180),
        "criticality": row["criticality"],
        "days_since_temp_fix": row["days_since_temp_fix"],
        "predicted_lifespan": row["predicted_temp_lifespan_days"],
        "temp_fix_age_ratio": row["temp_fix_age_ratio"],
        "estimated_repair_time": row["estimated_repair_time_min"],
        "priority_score": row["priority_score"],
    }

    # Department one-hot
    for dept in DEPARTMENTS:
        safe = dept.replace(" ", "_").replace("&", "and")
        features[f"dept_{safe}"] = int(row["department"] == dept)

    # Interaction features
    features["age_x_traffic"] = row["temp_fix_age_ratio"] * row["criticality"]
    features["severity_x_age"] = row["severity"] * row["temp_fix_age_ratio"]
    features["urgency_score"] = row["temp_fix_age_ratio"] ** 2

    return features

feature_rows = df.apply(build_features, axis=1)
X = pd.DataFrame(feature_rows.tolist())
y = df["failed_within_7_days"].values

feature_cols = list(X.columns)
print(f"  Feature count: {len(feature_cols)}")
print(f"  Features: {feature_cols}")

# ============================================================
# 4. TRAIN XGBOOST
# ============================================================
print("\n[4/6] Training XGBoost classifier...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=42,
    eval_metric="logloss",
)

model.fit(X_train, y_train)
print("  ✓ Model trained")

# ============================================================
# 5. EVALUATE
# ============================================================
print("\n[5/6] Evaluating model...")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy":  round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
    "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
    "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
    "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
}

print("\n  📊 MODEL PERFORMANCE:")
for k, v in metrics.items():
    print(f"    {k:12s}: {v}")

print("\n  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
if cm.shape == (2, 2):
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

# Feature importance
importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n  Top 10 Feature Importances:")
print(importances.head(10).to_string(index=False))

# ============================================================
# 6. SAVE
# ============================================================
print("\n[6/6] Saving model...")

os.makedirs("src/models/saved", exist_ok=True)
os.makedirs("output/reports", exist_ok=True)

with open("src/models/saved/temp_fix_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("src/models/saved/temp_fix_features.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

# Save report
with open("output/reports/temp_fix_model_performance.txt", "w", encoding="utf-8") as f:
    f.write("TEMP-FIX FAILURE PREDICTOR — PERFORMANCE REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Training rows: {len(X_train)}\n")
    f.write(f"Test rows:     {len(X_test)}\n")
    f.write(f"Positive rate: {y.mean():.2%}\n\n")
    f.write("METRICS:\n")
    for k, v in metrics.items():
        f.write(f"  {k:12s}: {v}\n")
    f.write("\nTOP 15 FEATURE IMPORTANCES:\n")
    f.write(importances.head(15).to_string(index=False))

print(f"  ✓ Model saved:    src/models/saved/temp_fix_model.pkl")
print(f"  ✓ Features saved: src/models/saved/temp_fix_features.pkl")
print(f"  ✓ Report saved:   output/reports/temp_fix_model_performance.txt")
print(f"\n  🎯 ROC-AUC: {metrics['roc_auc']}  |  F1: {metrics['f1']}")
print(f"\n{'=' * 70}")
print("STAGE 2 COMPLETE — Ready for Stage 3 (Priority Escalation)")
print("=" * 70)