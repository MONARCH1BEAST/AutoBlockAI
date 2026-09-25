"""
src/utils/add_temp_fix_data.py

Enriches maintenance_tasks.csv with realistic temporary-fix fields:
  - temporary_fix_applied   : bool
  - temporary_fix_type      : str
  - temporary_fix_applied_at: datetime (relative to a reference date)
  - predicted_temp_lifespan_days : int (ground truth, used as ML target)
  - days_since_temp_fix     : int

Output:
  - data/processed/maintenance_tasks_enriched.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

print("=" * 70)
print("STAGE 1: ADDING TEMPORARY FIX DATA")
print("=" * 70)

# ============================================================
# 1. LOAD TASKS
# ============================================================
print("\n[1/4] Loading maintenance tasks...")
tasks = pd.read_csv("data/processed/maintenance_tasks.csv")
print(f"  Loaded {len(tasks)} tasks")

# Reference date — assume this is "today" for the demo
REFERENCE_DATE = datetime(2026, 9, 24)
print(f"  Reference date: {REFERENCE_DATE.date()}")

# ============================================================
# 2. DEFINE TEMP FIX TYPES PER DEPARTMENT
# ============================================================
print("\n[2/4] Assigning temp fix types and lifespans...")

TEMP_FIX_TYPES = {
    "Engineering": [
        ("Rail Joint Clamp",        {"mean": 10, "std": 4}),
        ("Fish Plate Emergency",    {"mean": 12, "std": 5}),
        ("Sleeper Temporary Support", {"mean": 7, "std": 3}),
        ("Ballast Patching",        {"mean": 14, "std": 6}),
        ("Gauge Tie Bar",           {"mean": 9, "std": 4}),
    ],
    "Signal & Telecom": [
        ("Relay Bypass",            {"mean": 3, "std": 1.5}),
        ("Point Machine Lock",      {"mean": 5, "std": 2}),
        ("Cable Jumper",            {"mean": 4, "std": 2}),
        ("Fuse Replacement",        {"mean": 6, "std": 2.5}),
        ("Track Circuit Override",  {"mean": 2, "std": 1}),
    ],
    "Traction Distribution": [
        ("OHE Jumper Wire",         {"mean": 6, "std": 3}),
        ("Insulator Bypass",        {"mean": 8, "std": 3}),
        ("Temporary Grounding",     {"mean": 4, "std": 2}),
        ("Bonding Jumper",          {"mean": 10, "std": 4}),
        ("Section Insulator Bridge", {"mean": 7, "std": 3}),
    ],
}

# ============================================================
# 3. ASSIGN TEMP-FIX STATUS
# ============================================================
# Realistic rule: higher severity → higher probability of temp fix
# (because a critical defect MUST be made safe quickly)

def temp_fix_probability(severity):
    """Higher severity = higher chance of temp fix."""
    mapping = {1: 0.05, 2: 0.15, 3: 0.35, 4: 0.60, 5: 0.85}
    return mapping.get(severity, 0.30)

print("  Computing temp-fix probability per task...")
tasks["temp_fix_prob"] = tasks["severity"].apply(temp_fix_probability)
tasks["temporary_fix_applied"] = np.random.random(len(tasks)) < tasks["temp_fix_prob"]
tasks = tasks.drop(columns=["temp_fix_prob"])

num_with_temp = tasks["temporary_fix_applied"].sum()
print(f"  Tasks with temp fix: {num_with_temp} ({num_with_temp/len(tasks)*100:.1f}%)")

# ============================================================
# 4. GENERATE TEMP-FIX DETAILS
# ============================================================
print("\n[3/4] Generating temp-fix details...")

def generate_temp_fix(row):
    if not row["temporary_fix_applied"]:
        return pd.Series({
            "temporary_fix_type": None,
            "temporary_fix_applied_at": None,
            "predicted_temp_lifespan_days": None,
            "days_since_temp_fix": None,
        })

    dept = row["department"]
    fix_types = TEMP_FIX_TYPES.get(dept, TEMP_FIX_TYPES["Engineering"])
    fix_type, lifespan_params = fix_types[np.random.randint(len(fix_types))]

    # Generate lifespan using truncated normal (min 1 day)
    lifespan = max(1, int(np.random.normal(
        lifespan_params["mean"], lifespan_params["std"]
    )))

    # Temp fix was applied somewhere between 0 and lifespan days ago
    # Bias toward "recently applied" for lower severity, "aging" for higher
    max_days_ago = max(1, int(lifespan * 0.9))
    days_ago = np.random.randint(0, max_days_ago + 1)

    applied_at = REFERENCE_DATE - timedelta(days=days_ago)

    return pd.Series({
        "temporary_fix_type": fix_type,
        "temporary_fix_applied_at": applied_at,
        "predicted_temp_lifespan_days": lifespan,
        "days_since_temp_fix": days_ago,
    })

# Apply to all rows
temp_fix_data = tasks.apply(generate_temp_fix, axis=1)
tasks = pd.concat([tasks, temp_fix_data], axis=1)

# Compute "temp fix age ratio" — 0 = fresh, 1 = about to fail
tasks["temp_fix_age_ratio"] = tasks.apply(
    lambda r: (r["days_since_temp_fix"] / r["predicted_temp_lifespan_days"])
    if r["temporary_fix_applied"] and r["predicted_temp_lifespan_days"] > 0
    else 0.0,
    axis=1
).round(3)

# ============================================================
# 5. SAVE
# ============================================================
print("\n[4/4] Saving enriched tasks...")

output_path = "data/processed/maintenance_tasks_enriched.csv"
tasks.to_csv(output_path, index=False)

# ============================================================
# 6. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("TEMP-FIX ENRICHMENT COMPLETE")
print("=" * 70)

print(f"\n📊 SUMMARY:")
print(f"  Total tasks:            {len(tasks)}")
print(f"  With temp fix:          {int(tasks['temporary_fix_applied'].sum())}")
print(f"  Without temp fix:       {int((~tasks['temporary_fix_applied']).sum())}")

with_temp = tasks[tasks["temporary_fix_applied"]]

print(f"\n  Temp fix types (top 10):")
top_types = with_temp["temporary_fix_type"].value_counts().head(10)
for t, c in top_types.items():
    print(f"    {t:35s} {c}")

print(f"\n  Lifespan stats (days):")
print(f"    Mean:   {with_temp['predicted_temp_lifespan_days'].mean():.1f}")
print(f"    Median: {with_temp['predicted_temp_lifespan_days'].median():.1f}")
print(f"    Min:    {with_temp['predicted_temp_lifespan_days'].min()}")
print(f"    Max:    {with_temp['predicted_temp_lifespan_days'].max()}")

print(f"\n  ⚠️  Temp fixes approaching failure (>80% of lifespan used):")
critical_temp = with_temp[with_temp["temp_fix_age_ratio"] > 0.8]
print(f"    Count: {len(critical_temp)}")
if len(critical_temp) > 0:
    print(f"    Sample:")
    print(critical_temp[[
        "task_id", "department", "temporary_fix_type",
        "days_since_temp_fix", "predicted_temp_lifespan_days", "temp_fix_age_ratio"
    ]].head(5).to_string(index=False))

print(f"\n  ✓ Saved {output_path}")
print(f"\n{'=' * 70}")
print("STAGE 1 COMPLETE — Ready for Stage 2 (ML model training)")
print("=" * 70)