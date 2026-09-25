"""
src/models/baseline_comparison.py

Compares XGBoost against simple rule-based baselines to prove
that ML adds real value over naive strategies.

Baselines:
  1. Random       — pick any valid window at random
  2. LongestWindow — always pick the longest available window
  3. PriorityMatch — always pick the window with highest priority task fit
  4. SeverityOnly  — always pick highest severity task

Output:
  - output/reports/baseline_comparison.txt
  - output/reports/baseline_comparison.csv
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier

np.random.seed(42)

print("=" * 70)
print("BASELINE COMPARISON — XGBoost vs Rule-Based Heuristics")
print("=" * 70)

# ============================================================
# 1. REBUILD TRAINING DATA
# ============================================================
print("\n[1/5] Rebuilding training data...")

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
X = df_train[feature_cols]
y = df_train["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training rows: {len(df_train)}")
print(f"  Test rows: {len(X_test)}")

# ============================================================
# 2. TRAIN XGBOOST
# ============================================================
print("\n[2/5] Training XGBoost...")

scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
model = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=42, eval_metric="logloss",
)
model.fit(X_train, y_train)
xgb_proba = model.predict_proba(X_test)[:, 1]
xgb_pred = model.predict(X_test)

# ============================================================
# 3. DEFINE BASELINES
# ============================================================
print("\n[3/5] Computing baselines...")

# Random
random_proba = np.random.random(len(X_test))
random_pred = (random_proba > 0.5).astype(int)

# Longest Window
longest_proba = (X_test["window_length"] - X_test["window_length"].min()) / \
                (X_test["window_length"].max() - X_test["window_length"].min() + 1e-9)
longest_pred = (longest_proba > 0.5).astype(int)

# Priority Match
priority_proba = X_test["priority_score"] * X_test["window_has_capacity"]
priority_proba = (priority_proba - priority_proba.min()) / \
                 (priority_proba.max() - priority_proba.min() + 1e-9)
priority_pred = (priority_proba > 0.5).astype(int)

# Severity Only
severity_proba = X_test["severity"] / 5.0
severity_pred = (severity_proba > 0.5).astype(int)

# ============================================================
# 4. EVALUATE ALL
# ============================================================
print("\n[4/5] Evaluating...")

def evaluate(name, y_true, y_pred, y_proba):
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_proba), 4),
    }

results = [
    evaluate("[RANDOM] Random Baseline",  y_test, random_pred,   random_proba),
    evaluate("[LONGEST] Longest Window",  y_test, longest_pred,  longest_proba),
    evaluate("[PRIORITY] Priority Match", y_test, priority_pred, priority_proba),
    evaluate("[SEVERITY] Severity Only",  y_test, severity_pred, severity_proba),
    evaluate("[XGBOOST] XGBoost (Ours)",  y_test, xgb_pred,      xgb_proba),
]

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)

print("\n" + "=" * 70)
print("BASELINE COMPARISON RESULTS")
print("=" * 70)
print(results_df.to_string(index=False))

best_baseline = results_df[results_df["Model"] != "[XGBOOST] XGBoost (Ours)"].iloc[0]
xgb_row = results_df[results_df["Model"] == "[XGBOOST] XGBoost (Ours)"].iloc[0]
improvement = (xgb_row["ROC-AUC"] - best_baseline["ROC-AUC"]) / best_baseline["ROC-AUC"] * 100

print(f"\n  XGBoost vs best baseline:")
print(f"     Best baseline:  {best_baseline['Model']} - ROC-AUC {best_baseline['ROC-AUC']}")
print(f"     XGBoost:        ROC-AUC {xgb_row['ROC-AUC']}")
print(f"     Relative lift:  +{improvement:.1f}%")

# ============================================================
# 5. SAVE
# ============================================================
print("\n[5/5] Saving report...")

os.makedirs("output/reports", exist_ok=True)

results_df.to_csv("output/reports/baseline_comparison.csv", index=False)

with open("output/reports/baseline_comparison.txt", "w", encoding="utf-8") as f:
    f.write("ML BASELINE COMPARISON REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Test set size: {len(X_test)} rows\n")
    f.write(f"Positive class rate: {y_test.mean():.2%}\n\n")
    f.write("RESULTS:\n")
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nXGBoost lift over best baseline: +{improvement:.1f}%\n")
    f.write(f"Best baseline: {best_baseline['Model']}\n")
    f.write(f"XGBoost ROC-AUC: {xgb_row['ROC-AUC']}\n")

print(f"  Saved output/reports/baseline_comparison.csv")
print(f"  Saved output/reports/baseline_comparison.txt")
print(f"\n{'=' * 70}")