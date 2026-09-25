"""
src/models/train_assignment_model.py

Train an XGBoost model that predicts whether a (task, window) pair
is a good assignment.

Key design (v2 - fixed data leakage):
  Negatives are built using the SAME task assigned to WRONG windows,
  so the model cannot cheat by relying on severity. It must learn from
  window features and interaction features.

Input:
  - data/processed/maintenance_tasks.csv
  - data/processed/corridor_availability_week.csv
  - data/processed/train_delay_stats.csv
  - output/schedules/scheduled_blocks.csv  (ground-truth positives)

Output:
  - src/models/saved/assignment_model.pkl
  - src/models/saved/feature_columns.pkl
  - output/reports/model_performance.txt
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

np.random.seed(42)

print("=" * 70)
print("TRAINING ASSIGNMENT ML MODEL")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/6] Loading data...")

tasks = pd.read_csv("data/processed/maintenance_tasks.csv")
windows = pd.read_csv("data/processed/corridor_availability_week.csv")
delays = pd.read_csv("data/processed/train_delay_stats.csv")
schedule = pd.read_csv("output/schedules/scheduled_blocks.csv")

# Clean and attach delay stats to windows
delays["std_delay_min"] = delays["std_delay_min"].clip(upper=60)
delays["avg_delay_min"] = delays["avg_delay_min"].clip(upper=90)
windows = windows.merge(
    delays[["corridor_id", "avg_delay_min", "std_delay_min"]],
    on="corridor_id", how="left"
).fillna({"avg_delay_min": 20, "std_delay_min": 15})

print(f"  Tasks:     {len(tasks)}")
print(f"  Windows:   {len(windows)}")
print(f"  Positives: {len(schedule)}")

# ============================================================
# 2. BUILD FEATURE ENGINEERING FUNCTION
# ============================================================
print("\n[2/6] Engineering features...")

DEPARTMENTS = ["Engineering", "Signal & Telecom", "Traction Distribution"]

def build_features(task_row, window_row):
    """Build feature vector for a (task, window) pair."""
    repair_time = task_row["estimated_repair_time_min"]
    window_len = window_row["window_length_min"]
    start_min = window_row["window_start_min"]
    hour = (start_min / 60) % 24

    # --- Task features ---
    features = {
        "severity": task_row["severity"],
        "days_overdue": min(task_row["days_overdue"], 180),
        "repair_time": repair_time,
        "priority_score": task_row["priority_score"],
        "criticality": task_row["criticality"],
    }

    # Department one-hot
    for dept in DEPARTMENTS:
        safe_name = dept.replace(" ", "_").replace("&", "and")
        features[f"dept_{safe_name}"] = int(task_row["department"] == dept)

    # --- Window features ---
    features["window_length"] = window_len
    features["window_start_hour"] = hour
    features["day_of_week"] = window_row["day_of_week"]
    features["corridor_criticality"] = window_row["criticality"]
    features["corridor_avg_delay"] = window_row["avg_delay_min"]
    features["corridor_std_delay"] = window_row["std_delay_min"]

    # --- Interaction features ---
    features["repair_time_fit"] = 1.0 - (repair_time / window_len)
    features["window_has_capacity"] = int(window_len >= repair_time)
    features["priority_x_criticality"] = task_row["priority_score"] * task_row["criticality"]
    features["severity_x_criticality"] = task_row["severity"] * task_row["criticality"]
    features["same_corridor"] = int(task_row["corridor_id"] == window_row["corridor_id"])

    # Cyclical time-of-day
    features["tod_sin"] = np.sin(2 * np.pi * hour / 24)
    features["tod_cos"] = np.cos(2 * np.pi * hour / 24)

    return features

# ============================================================
# 3. GENERATE POSITIVE + NEGATIVE TRAINING EXAMPLES (v2 - no leakage)
# ============================================================
print("\n[3/6] Generating training examples...")

# --- POSITIVES: from actual schedule output ---
positives = []
positive_pairs = set()  # dedup

for _, s in schedule.iterrows():
    task_rows = tasks[tasks["task_id"] == s["task_id"]]
    window_rows = windows[windows["window_id"] == s["window_id"]]
    if len(task_rows) == 0 or len(window_rows) == 0:
        continue
    task = task_rows.iloc[0]
    window = window_rows.iloc[0]
    f = build_features(task, window)
    f["label"] = 1
    positives.append(f)
    positive_pairs.add((s["task_id"], s["window_id"]))

print(f"  Positive examples: {len(positives)}")

# --- NEGATIVES: SAME task, WRONG window (no leakage) ---
NEGATIVES_PER_POSITIVE = 5
negatives = []
negative_seen = set()

for _, s in schedule.iterrows():
    task_rows = tasks[tasks["task_id"] == s["task_id"]]
    if len(task_rows) == 0:
        continue
    task = task_rows.iloc[0]

    # Candidate windows
    same_corridor = windows[windows["corridor_id"] == task["corridor_id"]]
    other_corridor = windows[windows["corridor_id"] != task["corridor_id"]]

    # 70% hard negatives (same corridor), 30% easy negatives (different corridor)
    n_hard = int(NEGATIVES_PER_POSITIVE * 0.7)
    n_easy = NEGATIVES_PER_POSITIVE - n_hard

    sampled = []
    if len(same_corridor) > 0:
        sampled += same_corridor.sample(
            n=min(n_hard, len(same_corridor)),
            replace=False,
            random_state=int(hash(s["task_id"]) % 10000)
        ).to_dict("records")
    if len(other_corridor) > 0:
        sampled += other_corridor.sample(
            n=min(n_easy, len(other_corridor)),
            replace=False,
            random_state=int(hash(s["task_id"]) % 10000) + 1
        ).to_dict("records")

    for window in sampled:
        pair = (task["task_id"], window["window_id"])
        if pair in positive_pairs or pair in negative_seen:
            continue
        f = build_features(task, window)
        f["label"] = 0
        negatives.append(f)
        negative_seen.add(pair)

print(f"  Negative examples: {len(negatives)}")

# Combine and shuffle
df_train = pd.DataFrame(positives + negatives).sample(
    frac=1, random_state=42
).reset_index(drop=True)

print(f"  Total training rows: {len(df_train)}")
print(f"  Class balance: {df_train['label'].mean():.2%} positive")

# ---- Sanity check: no leakage ----
print(f"\n  🔍 LEAKAGE CHECK:")
pos_sev = df_train[df_train['label'] == 1]['severity'].mean()
neg_sev = df_train[df_train['label'] == 0]['severity'].mean()
print(f"     Avg severity (positives): {pos_sev:.2f}")
print(f"     Avg severity (negatives): {neg_sev:.2f}")
if abs(pos_sev - neg_sev) < 0.3:
    print(f"     ✅ No leakage — severities are similar")
else:
    print(f"     ⚠️  WARNING: Severities differ — check negative generation")

# ============================================================
# 4. TRAIN XGBOOST
# ============================================================
print("\n[4/6] Training XGBoost classifier...")

feature_cols = [c for c in df_train.columns if c != "label"]
X = df_train[feature_cols]
y = df_train["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance
scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)

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
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

print("\n  Top 10 Feature Importances:")
importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print(importances.head(10).to_string(index=False))

# ============================================================
# 6. SAVE MODEL + REPORT
# ============================================================
print("\n[6/6] Saving model and report...")

os.makedirs("src/models/saved", exist_ok=True)
os.makedirs("output/reports", exist_ok=True)

with open("src/models/saved/assignment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("src/models/saved/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

# Save report
with open("output/reports/model_performance.txt", "w") as f:
    f.write("ASSIGNMENT ML MODEL — PERFORMANCE REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Training rows: {len(df_train)}\n")
    f.write(f"Positive: {int(y.sum())} | Negative: {int((y == 0).sum())}\n\n")
    f.write(f"Avg severity (positives): {pos_sev:.2f}\n")
    f.write(f"Avg severity (negatives): {neg_sev:.2f}\n")
    f.write("(Close values indicate no data leakage)\n\n")
    f.write("METRICS:\n")
    for k, v in metrics.items():
        f.write(f"  {k:12s}: {v}\n")
    f.write("\nFEATURE IMPORTANCE (Top 15):\n")
    f.write(importances.head(15).to_string(index=False))

print(f"\n  ✓ Model saved:    src/models/saved/assignment_model.pkl")
print(f"  ✓ Features saved: src/models/saved/feature_columns.pkl")
print(f"  ✓ Report saved:   output/reports/model_performance.txt")
print(f"\n  🎯 ROC-AUC: {metrics['roc_auc']}  |  F1: {metrics['f1']}")
print(f"\n{'=' * 70}")