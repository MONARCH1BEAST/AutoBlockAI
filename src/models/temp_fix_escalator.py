"""
src/models/temp_fix_escalator.py

Uses the trained temp-fix failure model to escalate task priorities
dynamically.

Output:
  - data/processed/maintenance_tasks_escalated.csv
"""

import pandas as pd
import numpy as np
import pickle
import os

print("=" * 70)
print("STAGE 3: PRIORITY ESCALATION ENGINE")
print("=" * 70)

# ============================================================
# 1. LOAD MODEL AND DATA
# ============================================================
print("\n[1/4] Loading model and enriched tasks...")

with open("src/models/saved/temp_fix_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("src/models/saved/temp_fix_features.pkl", "rb") as f:
    feature_cols = pickle.load(f)

tasks = pd.read_csv("data/processed/maintenance_tasks_enriched.csv")
print(f"  Loaded {len(tasks)} tasks")
print(f"  Model ready with {len(feature_cols)} features")

# ============================================================
# 2. BUILD FEATURES (same as training)
# ============================================================
print("\n[2/4] Computing failure risk for every task...")

DEPARTMENTS = ["Engineering", "Signal & Telecom", "Traction Distribution"]

def build_features(row):
    features = {
        "severity": row["severity"],
        "days_overdue": min(row["days_overdue"], 180),
        "criticality": row["criticality"],
        "days_since_temp_fix": row["days_since_temp_fix"] if pd.notna(row["days_since_temp_fix"]) else 0,
        "predicted_lifespan": row["predicted_temp_lifespan_days"] if pd.notna(row["predicted_temp_lifespan_days"]) else 0,
        "temp_fix_age_ratio": row["temp_fix_age_ratio"] if pd.notna(row["temp_fix_age_ratio"]) else 0,
        "estimated_repair_time": row["estimated_repair_time_min"],
        "priority_score": row["priority_score"],
    }
    for dept in DEPARTMENTS:
        safe = dept.replace(" ", "_").replace("&", "and")
        features[f"dept_{safe}"] = int(row["department"] == dept)

    features["age_x_traffic"] = features["temp_fix_age_ratio"] * row["criticality"]
    features["severity_x_age"] = row["severity"] * features["temp_fix_age_ratio"]
    features["urgency_score"] = features["temp_fix_age_ratio"] ** 2
    return features

# ============================================================
# 3. COMPUTE FAILURE RISK AND ESCALATED PRIORITY
# ============================================================
# Only apply ML model to tasks WITH temp fixes.
# Tasks without temp fixes keep their base priority.

has_temp = tasks["temporary_fix_applied"] == True

print(f"  Tasks with temp fixes: {has_temp.sum()}")
print(f"  Tasks without temp fixes: {(~has_temp).sum()}")

# Default values
tasks["temp_failure_risk"] = 0.0
tasks["escalated_priority"] = tasks["priority_score"]
tasks["escalation_delta"] = 0.0

if has_temp.sum() > 0:
    # Build features for temp-fix tasks only
    temp_df = tasks[has_temp].copy()
    feature_rows = temp_df.apply(build_features, axis=1)
    X = pd.DataFrame(feature_rows.tolist())[feature_cols]

    # Predict failure risk
    risk = model.predict_proba(X)[:, 1]
    temp_df["temp_failure_risk"] = risk

    # --------------------------------------------------------
    # ESCALATION FORMULA
    # --------------------------------------------------------
    # escalated = base + (1 - base) * risk^strength
    #
    # effect:
    #   risk=0.05 → tiny lift
    #   risk=0.50 → moderate lift
    #   risk=0.90 → near 1.0
    ESCALATION_STRENGTH = 0.8

    base = temp_df["priority_score"]
    escalation_factor = risk ** ESCALATION_STRENGTH
    temp_df["escalated_priority"] = np.minimum(
        1.0,
        base + (1 - base) * escalation_factor
    )
    temp_df["escalation_delta"] = (
        temp_df["escalated_priority"] - temp_df["priority_score"]
    ).round(3)

    # Write back
    tasks.loc[has_temp, "temp_failure_risk"] = temp_df["temp_failure_risk"].round(4)
    tasks.loc[has_temp, "escalated_priority"] = temp_df["escalated_priority"].round(4)
    tasks.loc[has_temp, "escalation_delta"] = temp_df["escalation_delta"]

# ============================================================
# 4. SAVE + SUMMARY
# ============================================================
print("\n[3/4] Saving escalated tasks...")
output_path = "data/processed/maintenance_tasks_escalated.csv"
tasks.to_csv(output_path, index=False)

print("\n[4/4] Summary\n")
print("=" * 70)
print("PRIORITY ESCALATION SUMMARY")
print("=" * 70)

escalated_df = tasks[has_temp].copy()

print(f"\n  Tasks with temp fixes: {len(escalated_df)}")
print(f"\n  Base priority stats (temp-fix tasks):")
print(f"    Mean:   {escalated_df['priority_score'].mean():.3f}")
print(f"    Max:    {escalated_df['priority_score'].max():.3f}")

print(f"\n  Escalated priority stats:")
print(f"    Mean:   {escalated_df['escalated_priority'].mean():.3f}")
print(f"    Max:    {escalated_df['escalated_priority'].max():.3f}")
print(f"    Lifted: {(escalated_df['escalation_delta'] > 0).sum()} tasks")

print(f"\n  Escalation buckets:")
buckets = pd.cut(
    escalated_df["escalation_delta"],
    bins=[-0.001, 0.01, 0.05, 0.15, 0.30, 1.0],
    labels=["No lift", "Small (<0.05)", "Moderate (0.05-0.15)", "Large (0.15-0.30)", "Critical (>0.30)"]
)
print(buckets.value_counts().sort_index().to_string())

print(f"\n  🚨 TOP 10 ESCALATED TASKS (highest failure risk):")
top = escalated_df.nlargest(10, "temp_failure_risk")[[
    "task_id", "department", "temporary_fix_type",
    "days_since_temp_fix", "predicted_temp_lifespan_days",
    "priority_score", "escalated_priority", "temp_failure_risk"
]]
print(top.to_string(index=False))

print(f"\n  ✓ Saved {output_path}")
print(f"\n{'=' * 70}")
print("STAGE 3 COMPLETE — Ready for Stage 4 (Updated Optimizer)")
print("=" * 70)