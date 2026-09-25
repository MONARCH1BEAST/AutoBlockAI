"""
src/models/ml_optimizer.py (v6 - FINAL)

Final optimizer with:
  - Corridor-first sorting: expired temp fixes processed first
  - ML-driven assignment with score boosts for critical/expired tasks
  - Golden Block clustering preserved
  - Monte Carlo Regret Simulation
  - Complete temp-fix risk reporting

Outputs:
  - output/schedules/ml_scheduled_blocks_v6.csv
  - output/schedules/ml_golden_blocks_v6.csv
  - output/reports/temp_fix_risk_report_v6.csv
  - output/reports/corridor_summary_v6.csv
"""

import pandas as pd
import numpy as np
import pickle
from scipy import stats as scipy_stats

print("=" * 70)
print("STAGE 4: FINAL OPTIMIZER (v6 - CORRIDOR-FIRST PRIORITIZATION)")
print("=" * 70)

# ============================================================
# 1. LOAD MODEL + DATA
# ============================================================
print("\n[1/6] Loading model and escalated tasks...")

with open("src/models/saved/assignment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("src/models/saved/feature_columns.pkl", "rb") as f:
    feature_cols = pickle.load(f)

tasks = pd.read_csv("data/processed/maintenance_tasks_escalated.csv")
windows = pd.read_csv("data/processed/corridor_availability_week.csv")
delays = pd.read_csv("data/processed/train_delay_stats.csv")
corridors = pd.read_csv("data/processed/corridor_master.csv")

delays["std_delay_min"] = delays["std_delay_min"].clip(upper=60)
delays["avg_delay_min"] = delays["avg_delay_min"].clip(upper=90)

windows = windows.merge(
    delays[["corridor_id", "avg_delay_min", "std_delay_min"]],
    on="corridor_id", how="left"
).fillna({"avg_delay_min": 20, "std_delay_min": 15})

# Effective priority
tasks["effective_priority"] = tasks["escalated_priority"].fillna(tasks["priority_score"])

# ============================================================
# 2. COMPUTE SORTING KEY (this is the KEY fix)
# ============================================================
# Priority tiers:
#   Tier 0: Expired temp fix (age_ratio >= 1.0) → process FIRST
#   Tier 1: Critical temp fix (age_ratio > 0.8) → process SECOND
#   Tier 2: Aging temp fix (age_ratio > 0.5) → process THIRD
#   Tier 3: Everything else → process LAST
# Within each tier, sort by effective_priority DESC

def compute_tier(row):
    if row["temporary_fix_applied"] != True or pd.isna(row.get("temp_fix_age_ratio")):
        return 3
    ratio = row["temp_fix_age_ratio"]
    if ratio >= 1.0:
        return 0  # Expired — URGENT
    elif ratio > 0.8:
        return 1  # Critical
    elif ratio > 0.5:
        return 2  # Aging
    else:
        return 3  # Fresh or no temp fix

tasks["priority_tier"] = tasks.apply(compute_tier, axis=1)

print(f"\n  Task distribution by tier:")
tier_counts = tasks["priority_tier"].value_counts().sort_index()
tier_labels = {0: "Expired temp fix", 1: "Critical (80-100%)",
               2: "Aging (50-80%)", 3: "Fresh / no temp fix"}
for tier in sorted(tier_counts.index):
    print(f"    Tier {tier} ({tier_labels[tier]:25s}): {tier_counts[tier]}")

# ============================================================
# 3. FEATURE BUILDER
# ============================================================
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
        "priority_score": task_row["effective_priority"],
        "criticality": task_row["criticality"],
    }
    for dept in DEPARTMENTS:
        safe = dept.replace(" ", "_").replace("&", "and")
        features[f"dept_{safe}"] = int(task_row["department"] == dept)

    features["window_length"] = window_len
    features["window_start_hour"] = hour
    features["day_of_week"] = window_row["day_of_week"]
    features["corridor_criticality"] = window_row["criticality"]
    features["corridor_avg_delay"] = window_row["avg_delay_min"]
    features["corridor_std_delay"] = window_row["std_delay_min"]

    features["repair_time_fit"] = 1.0 - (repair_time / window_len)
    features["window_has_capacity"] = int(window_len >= repair_time)
    features["priority_x_criticality"] = task_row["effective_priority"] * task_row["criticality"]
    features["severity_x_criticality"] = task_row["severity"] * task_row["criticality"]
    features["same_corridor"] = int(task_row["corridor_id"] == window_row["corridor_id"])
    features["tod_sin"] = np.sin(2 * np.pi * hour / 24)
    features["tod_cos"] = np.cos(2 * np.pi * hour / 24)
    return features

# ============================================================
# 4. PER-CORRIDOR ML OPTIMIZATION
# ============================================================
print("\n[2/6] Running per-corridor ML assignment...")

TASKS_PER_CORRIDOR = 30
MAX_TASKS_PER_WINDOW = 6

all_assignments = []
corridor_summary = []
skipped_due_to_constraint = []

for _, corridor in corridors.iterrows():
    cid = corridor["corridor_id"]

    # Get top tasks for this corridor, sorted by tier then priority
    corridor_tasks = tasks[tasks["corridor_id"] == cid].copy()
    corridor_tasks = corridor_tasks.sort_values(
        ["priority_tier", "effective_priority"],
        ascending=[True, False]
    ).head(TASKS_PER_CORRIDOR)

    corridor_windows = windows[windows["corridor_id"] == cid].copy()

    if len(corridor_tasks) == 0 or len(corridor_windows) == 0:
        continue

    corridor_windows["assigned_count"] = 0
    corridor_windows["departments_in_window"] = corridor_windows.apply(lambda _: set(), axis=1)

    assigned_count = 0

    for _, task in corridor_tasks.iterrows():
        has_temp = task["temporary_fix_applied"] == True
        tier = task["priority_tier"]

        # Score all candidate windows
        candidates = []
        for w_idx, window in corridor_windows.iterrows():
            if window["window_length_min"] < task["estimated_repair_time_min"]:
                continue
            if window["assigned_count"] >= MAX_TASKS_PER_WINDOW:
                continue

            depts_in_window = window["departments_in_window"]
            same_dept_count = sum(1 for d in depts_in_window if d == task["department"])
            if same_dept_count >= 2:
                continue

            features = build_features(task, window)
            X = pd.DataFrame([features])[feature_cols]
            ml_prob = model.predict_proba(X)[0, 1]

            # Golden Block bonus
            other_depts = depts_in_window - {task["department"]}
            golden_bonus = 0.15 * len(other_depts)

            final_score = ml_prob + golden_bonus

            # TIER-BASED SCORE BOOST (this replaces the pre-pass)
            if tier == 0:      # expired
                final_score += 0.50
            elif tier == 1:    # critical
                final_score += 0.25
            elif tier == 2:    # aging
                final_score += 0.10

            candidates.append((w_idx, ml_prob, final_score))

        if not candidates:
            if has_temp:
                skipped_due_to_constraint.append({
                    "task_id": task["task_id"],
                    "corridor_id": cid,
                    "department": task["department"],
                    "temp_fix_type": task["temporary_fix_type"],
                    "temp_fix_age_ratio": task["temp_fix_age_ratio"],
                    "priority_tier": tier,
                    "reason": "no_valid_window",
                })
            continue

        best_w_idx, ml_prob, final_score = max(candidates, key=lambda x: x[2])
        best_window = corridor_windows.loc[best_w_idx]

        # Tag assignment type
        if tier == 0:
            assign_type = "EXPIRED_TEMP_FIX"
        elif tier == 1:
            assign_type = "CRITICAL_TEMP_FIX"
        elif tier == 2:
            assign_type = "AGING_TEMP_FIX"
        else:
            assign_type = "ML_OPTIMIZED"

        all_assignments.append({
            "task_id": task["task_id"],
            "window_id": best_window["window_id"],
            "corridor_id": cid,
            "corridor_name": corridor["corridor_name"],
            "department": task["department"],
            "defect_type": task["defect_type"],
            "severity": task["severity"],
            "estimated_repair_time_min": task["estimated_repair_time_min"],
            "base_priority": task["priority_score"],
            "effective_priority": task["effective_priority"],
            "has_temp_fix": has_temp,
            "temp_fix_type": task["temporary_fix_type"] if has_temp else None,
            "temp_fix_age_ratio": task["temp_fix_age_ratio"] if has_temp else None,
            "temp_failure_risk": task["temp_failure_risk"] if has_temp else None,
            "priority_tier": tier,
            "day_of_week": best_window["day_of_week"],
            "window_start_min": best_window["window_start_min"],
            "window_end_min": best_window["window_end_min"],
            "window_length_min": best_window["window_length_min"],
            "ml_probability": round(ml_prob, 4),
            "final_score": round(final_score, 4),
            "assignment_type": assign_type,
        })

        corridor_windows.at[best_w_idx, "assigned_count"] += 1
        corridor_windows.at[best_w_idx, "departments_in_window"] = (
            best_window["departments_in_window"] | {task["department"]}
        )
        assigned_count += 1

    corridor_summary.append({
        "corridor_id": cid,
        "corridor_name": corridor["corridor_name"],
        "tasks_available": len(corridor_tasks),
        "tasks_assigned": assigned_count,
        "expired_temp_fixes": int((corridor_tasks["priority_tier"] == 0).sum()),
        "critical_temp_fixes": int((corridor_tasks["priority_tier"] == 1).sum()),
    })

schedule_df = pd.DataFrame(all_assignments)
print(f"  Total assignments:      {len(schedule_df)}")
print(f"  Expired temp fixes:     {int((schedule_df['assignment_type'] == 'EXPIRED_TEMP_FIX').sum())}")
print(f"  Critical temp fixes:    {int((schedule_df['assignment_type'] == 'CRITICAL_TEMP_FIX').sum())}")
print(f"  Aging temp fixes:       {int((schedule_df['assignment_type'] == 'AGING_TEMP_FIX').sum())}")
print(f"  ML-optimized:           {int((schedule_df['assignment_type'] == 'ML_OPTIMIZED').sum())}")
print(f"  Corridors covered:      {schedule_df['corridor_id'].nunique()}")

# ============================================================
# 5. DETECT GOLDEN BLOCKS
# ============================================================
print("\n[3/6] Detecting Golden Blocks...")

golden_blocks = (
    schedule_df.groupby("window_id")
    .agg(
        corridor_id=("corridor_id", "first"),
        corridor_name=("corridor_name", "first"),
        day_of_week=("day_of_week", "first"),
        window_start_min=("window_start_min", "first"),
        window_end_min=("window_end_min", "first"),
        window_length_min=("window_length_min", "first"),
        num_tasks=("task_id", "count"),
        departments=("department", lambda x: sorted(set(x))),
        num_departments=("department", lambda x: len(set(x))),
        temp_fix_tasks=("has_temp_fix", "sum"),
        expired_fixes=("assignment_type", lambda x: (x == "EXPIRED_TEMP_FIX").sum()),
        critical_fixes=("assignment_type", lambda x: (x == "CRITICAL_TEMP_FIX").sum()),
        avg_escalated_priority=("effective_priority", "mean"),
        avg_ml_prob=("ml_probability", "mean"),
    )
    .reset_index()
)
golden_blocks["is_golden"] = golden_blocks["num_departments"] >= 2

# ============================================================
# 6. MONTE CARLO REGRET SIMULATION
# ============================================================
print("\n[4/6] Running Monte Carlo Regret Simulation...")

delay_lookup = delays.set_index("corridor_id")[["avg_delay_min", "std_delay_min"]].to_dict("index")

def clash_prob(duration, avg_d, std_d):
    if std_d <= 0:
        return 0.0
    return float(1 - scipy_stats.norm.cdf((duration - avg_d) / std_d))

for idx, b in golden_blocks.iterrows():
    st = delay_lookup.get(b["corridor_id"], {"avg_delay_min": 20, "std_delay_min": 15})
    p = clash_prob(b["window_length_min"], st["avg_delay_min"], st["std_delay_min"])
    golden_blocks.at[idx, "clash_probability"] = round(p, 3)
    golden_blocks.at[idx, "operationally_safe"] = p < 0.10

# ============================================================
# 7. TEMP-FIX RISK REPORT
# ============================================================
print("\n[5/6] Generating temp-fix risk report...")

all_temp_fixes = tasks[tasks["temporary_fix_applied"] == True]
critical_temp_tasks = all_temp_fixes[all_temp_fixes["temp_fix_age_ratio"] > 0.8]
expired_temp_tasks = all_temp_fixes[all_temp_fixes["temp_fix_age_ratio"] >= 1.0]

critical_scheduled = schedule_df[schedule_df["task_id"].isin(critical_temp_tasks["task_id"])]
expired_scheduled = schedule_df[schedule_df["task_id"].isin(expired_temp_tasks["task_id"])]

temp_risk_report = pd.DataFrame({
    "metric": [
        "Total temp-fix tasks",
        "Total temp-fix tasks scheduled",
        "Critical temp fixes (>80% aged)",
        "Critical temp fixes scheduled",
        "Critical temp fixes UNSCHEDULED",
        "Expired temp fixes (>=100%)",
        "Expired temp fixes scheduled",
        "Expired temp fixes UNSCHEDULED",
        "Coverage of critical fixes (%)",
        "Coverage of expired fixes (%)",
    ],
    "value": [
        len(all_temp_fixes),
        int(schedule_df["has_temp_fix"].sum()),
        len(critical_temp_tasks),
        len(critical_scheduled),
        len(critical_temp_tasks) - len(critical_scheduled),
        len(expired_temp_tasks),
        len(expired_scheduled),
        len(expired_temp_tasks) - len(expired_scheduled),
        round(len(critical_scheduled) / max(len(critical_temp_tasks), 1) * 100, 1),
        round(len(expired_scheduled) / max(len(expired_temp_tasks), 1) * 100, 1),
    ]
})

# ============================================================
# 8. SAVE
# ============================================================
print("\n[6/6] Saving outputs...")

schedule_df.to_csv("output/schedules/ml_scheduled_blocks_v6.csv", index=False)
golden_blocks.to_csv("output/schedules/ml_golden_blocks_v6.csv", index=False)
temp_risk_report.to_csv("output/reports/temp_fix_risk_report_v6.csv", index=False)
pd.DataFrame(corridor_summary).to_csv("output/reports/corridor_summary_v6.csv", index=False)

# ============================================================
# 9. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL OPTIMIZER RESULTS (v6 - CORRIDOR-FIRST PRIORITIZATION)")
print("=" * 70)

num_golden = golden_blocks["is_golden"].sum()
num_full = (golden_blocks["num_departments"] == 3).sum()
num_safe = golden_blocks["operationally_safe"].sum()
total_blocks = len(golden_blocks)
traditional = len(schedule_df)
reduction = (1 - total_blocks / traditional) * 100 if traditional > 0 else 0

print(f"\n📊 CORE METRICS:")
print(f"  Tasks assigned:            {len(schedule_df)}")
print(f"  Corridors covered:         {schedule_df['corridor_id'].nunique()}/20")
print(f"  Total blocks used:         {total_blocks}")
print(f"  Golden Blocks (2+ depts):  {num_golden}")
print(f"  Full Golden (3 depts):     {num_full}")
print(f"  Operationally safe:        {num_safe}/{total_blocks} ({num_safe/total_blocks*100:.0f}%)")

print(f"\n🎯 EFFICIENCY GAIN:")
print(f"  Traditional:               {traditional} blocks")
print(f"  Golden Block approach:     {total_blocks} blocks")
print(f"  Block reduction:           {reduction:.1f}%")

print(f"\n🔒 ASSIGNMENT BREAKDOWN:")
print(f"  Expired temp fixes:        {int((schedule_df['assignment_type'] == 'EXPIRED_TEMP_FIX').sum())}")
print(f"  Critical temp fixes:       {int((schedule_df['assignment_type'] == 'CRITICAL_TEMP_FIX').sum())}")
print(f"  Aging temp fixes:          {int((schedule_df['assignment_type'] == 'AGING_TEMP_FIX').sum())}")
print(f"  ML-optimized:              {int((schedule_df['assignment_type'] == 'ML_OPTIMIZED').sum())}")

print(f"\n🔧 TEMP-FIX AWARENESS:")
print(f"  Temp-fix tasks in schedule: {int(schedule_df['has_temp_fix'].sum())}")
print(f"  Critical temp fixes (>80%): {len(critical_temp_tasks)}")
print(f"  Critical scheduled:         {len(critical_scheduled)} ({len(critical_scheduled)/max(len(critical_temp_tasks),1)*100:.0f}%)")
print(f"  Expired temp fixes:         {len(expired_temp_tasks)}")
print(f"  Expired scheduled:          {len(expired_scheduled)} ({len(expired_scheduled)/max(len(expired_temp_tasks),1)*100:.0f}%)")

if len(skipped_due_to_constraint) > 0:
    print(f"\n  Tasks skipped (no valid window): {len(skipped_due_to_constraint)}")

print(f"\n🏆 TOP 5 GOLDEN BLOCKS:")
top_golden = golden_blocks[golden_blocks["is_golden"]].head(5)
for _, b in top_golden.iterrows():
    print(f"\n  {b['window_id']} | {b['corridor_name']}")
    print(f"    Tasks: {b['num_tasks']} | Departments: {b['departments']}")
    print(f"    Temp-fix tasks: {int(b['temp_fix_tasks'])} | Expired: {int(b['expired_fixes'])} | Critical: {int(b['critical_fixes'])}")
    print(f"    Avg escalated priority: {b['avg_escalated_priority']:.3f}")
    print(f"    Clash probability: {b['clash_probability']*100:.1f}%")

print(f"\n  ✓ output/schedules/ml_scheduled_blocks_v6.csv")
print(f"  ✓ output/schedules/ml_golden_blocks_v6.csv")
print(f"  ✓ output/reports/temp_fix_risk_report_v6.csv")
print(f"  ✓ output/reports/corridor_summary_v6.csv")
print(f"\n{'=' * 70}")
print("SYSTEM COMPLETE (v6)")
print("=" * 70)