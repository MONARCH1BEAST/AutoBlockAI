"""
src/models/cross_validation.py

5-fold cross-validation to verify model stability.

Output:
  - output/reports/cross_validation.txt
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import os

np.random.seed(42)

print("=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

# ============================================================
# 1. REBUILD TRAINING DATA
# ============================================================
print("\n[1/3] Rebuilding training data...")

tasks = pd.read_csv("data/processed/maintenance_tasks.csv")
windows = pd.read_csv("data/processed/corridor_availability_week.csv")
delays = pd.read_csv("data/processed/train_delay_stats.csv")
schedule = pd.read_csv("output/schedules/scheduled_blocks.csv")

delays["std_delay_min"] = delays["std_delay_min"].clip(upper=60)
delays["avg_delay_min"] = delays["avg_delay_min"].clip(upper=90)
windows = windows.merge(
    delays[["corridor_id", "avg_delay_min", "std_delay_min"]],
    on="corridor_id", how="left"
).fillna({"avg_delay_min": 20, "std_delay_min": 15})

DEPARTMENTS = ["Engineering", "Signal & Telecom", "Traction Distribution"]

def build_features(task_row, window_row):
    repair_time = task_row["estimated_repair_time_min"]
    window_len = window_row["window_length_min"]
    start_min = window_row["window_start_min"]
    hour = (start_min / 60) % 24

    features = {
        "severity": task_row["severity"],
        "days_overdue": min(task_row["days_overdue"], 180),
        "repair_time": repair_time,
        "priority_score": task_row["priority_score"],
        "criticality": task_row["criticality"],
    }
    for dept in DEPARTMENTS:
        safe_name = dept.replace(" ", "_").replace("&", "and")
        features[f"dept_{safe_name}"] = int(task_row["department"] == dept)

    features["window_length"] = window_len
    features["window_start_hour"] = hour
    features["day_of_week"] = window_row["day_of_week"]
    features["corridor_criticality"] = window_row["criticality"]
    features["corridor_avg_delay"] = window_row["avg_delay_min"]
    features["corridor_std_delay"] = window_row["std_delay_min"]

    features["repair_time_fit"] = 1.0 - (repair_time / window_len)
    features["window_has_capacity"] = int(window_len >= repair_time)
    features["priority_x_criticality"] = task_row["priority_score"] * task_row["criticality"]
    features["severity_x_criticality"] = task_row["severity"] * task_row["criticality"]
    features["same_corridor"] = int(task_row["corridor_id"] == window_row["corridor_id"])

    features["tod_sin"] = np.sin(2 * np.pi * hour / 24)
    features["tod_cos"] = np.cos(2 * np.pi * hour / 24)
    return features

# Positives
positives = []
positive_pairs = set()
for _, s in schedule.iterrows():
    task_rows = tasks[tasks["task_id"] == s["task_id"]]
    window_rows = windows[windows["window_id"] == s["window_id"]]
    if len(task_rows) == 0 or len(window_rows) == 0:
        continue
    f = build_features(task_rows.iloc[0], window_rows.iloc[0])
    f["label"] = 1
    positives.append(f)
    positive_pairs.add((s["task_id"], s["window_id"]))

# Negatives (no leakage)
NEGATIVES_PER_POSITIVE = 5
negatives = []
negative_seen = set()

for _, s in schedule.iterrows():
    task_rows = tasks[tasks["task_id"] == s["task_id"]]
    if len(task_rows) == 0:
        continue
    task = task_rows.iloc[0]
    same_corridor = windows[windows["corridor_id"] == task["corridor_id"]]
    other_corridor = windows[windows["corridor_id"] != task["corridor_id"]]

    n_hard = int(NEGATIVES_PER_POSITIVE * 0.7)
    n_easy = NEGATIVES_PER_POSITIVE - n_hard
    sampled = []
    if len(same_corridor) > 0:
        sampled += same_corridor.sample(
            n=min(n_hard, len(same_corridor)), replace=False,
            random_state=int(hash(s["task_id"]) % 10000)
        ).to_dict("records")
    if len(other_corridor) > 0:
        sampled += other_corridor.sample(
            n=min(n_easy, len(other_corridor)), replace=False,
            random_state=int(hash(s["task_id"]) % 10000) + 1
        ).to_dict("records")

    for w in sampled:
        pair = (task["task_id"], w["window_id"])
        if pair in positive_pairs or pair in negative_seen:
            continue
        f = build_features(task, w)
        f["label"] = 0
        negatives.append(f)
        negative_seen.add(pair)

df_train = pd.DataFrame(positives + negatives).sample(
    frac=1, random_state=42
).reset_index(drop=True)

feature_cols = [c for c in df_train.columns if c != "label"]
X = df_train[feature_cols].values
y = df_train["label"].values

print(f"  Training rows: {len(df_train)}")
print(f"  Features:      {len(feature_cols)}")
print(f"  Positive rate: {y.mean():.2%}")

# ============================================================
# 2. RUN 5-FOLD CV
# ============================================================
print("\n[2/3] Running 5-fold cross-validation...")

scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)
model = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=42, eval_metric="logloss",
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("  Computing ROC-AUC across 5 folds...")
auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

print("  Computing F1 across 5 folds...")
f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)

print("  Computing Accuracy across 5 folds...")
acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

# ============================================================
# 3. REPORT
# ============================================================
print("\n[3/3] Cross-Validation Results\n")

print("=" * 70)
print("5-FOLD CROSS-VALIDATION SUMMARY")
print("=" * 70)

metrics = {
    "ROC-AUC":  auc_scores,
    "F1":       f1_scores,
    "Accuracy": acc_scores,
}

for name, scores in metrics.items():
    print(f"\n  {name}:")
    print(f"    Fold scores: {[round(s, 4) for s in scores]}")
    print(f"    Mean:        {scores.mean():.4f}")
    print(f"    Std:         {scores.std():.4f}")
    print(f"    95% CI:      +/-{1.96 * scores.std():.4f}")

# Save
os.makedirs("output/reports", exist_ok=True)
with open("output/reports/cross_validation.txt", "w", encoding="utf-8") as f:
    f.write("5-FOLD CROSS-VALIDATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    for name, scores in metrics.items():
        f.write(f"{name}:\n")
        f.write(f"  Fold scores: {[round(s, 4) for s in scores]}\n")
        f.write(f"  Mean:        {scores.mean():.4f}\n")
        f.write(f"  Std:         {scores.std():.4f}\n")
        f.write(f"  95% CI:      +/-{1.96 * scores.std():.4f}\n\n")

print(f"\n  Saved output/reports/cross_validation.txt")
print(f"\n{'=' * 70}")