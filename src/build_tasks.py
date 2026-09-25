"""
src/build_tasks.py
Stage 2: Build unified maintenance tasks from Kaggle Failure dataset.

Input:
  - data/raw/kaggle_failure/indian_railway_predictive_maintenance_100k.csv
  - data/processed/corridor_master.csv

Output:
  - data/processed/maintenance_tasks.csv
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

print("=" * 70)
print("STAGE 2: BUILDING MAINTENANCE TASKS")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
print("\n[1/5] Loading Kaggle Failure dataset...")
# Use the 35-column version (has 3-department signals)
df = pd.read_csv("data/raw/kaggle_failure/indian_railway_predictive_maintenance_100k.csv")
print(f"  Loaded {len(df)} records")
print(f"  Failure types: {df['failure_type'].dropna().unique().tolist()}")

print("\n[2/5] Loading corridor master...")
corridors = pd.read_csv("data/processed/corridor_master.csv")
print(f"  Loaded {len(corridors)} corridors")

# ------------------------------------------------------------
# 2. Map failure_type → Department
# ------------------------------------------------------------
print("\n[3/5] Mapping failure types to departments...")

def map_department(failure_type):
    """Map a failure_type string to one of our three departments."""
    if pd.isna(failure_type):
        return None
    ft = str(failure_type).lower()
    if "track" in ft or "rail" in ft:
        return "Engineering"
    elif "signal" in ft or "point" in ft or "relay" in ft:
        return "Signal & Telecom"
    elif "bearing" in ft or "brake" in ft or "ohe" in ft or "traction" in ft or "motor" in ft:
        return "Traction Distribution"
    else:
        return "Engineering"  # default fallback

# Also use signal_system_status and ballast_condition as fallback for department
def infer_department_from_sensors(row):
    """If failure_type is missing, infer from sensor readings."""
    if pd.notna(row.get("failure_type")):
        return map_department(row["failure_type"])
    # Fallback: check signal status
    if row.get("signal_system_status") == "Faulty":
        return "Signal & Telecom"
    if row.get("ballast_condition") in ["Poor", "Degraded"]:
        return "Engineering"
    return "Traction Distribution"

df["department"] = df.apply(infer_department_from_sensors, axis=1)

# Filter to only rows with a clear failure (maintenance_required == 1)
df_defects = df[df["maintenance_required"] == 1].copy()
print(f"  Defects (maintenance_required == 1): {len(df_defects)}")
print(f"  Department distribution:\n{df_defects['department'].value_counts().to_string()}")

# ------------------------------------------------------------
# 3. Assign defects to corridors
# ------------------------------------------------------------
print("\n[4/5] Assigning defects to corridors...")

# Weight corridor assignment by criticality (busier corridors get more defects)
corridor_weights = corridors["criticality"] / corridors["criticality"].sum()

# Sample corridors for each defect
df_defects["corridor_id"] = np.random.choice(
    corridors["corridor_id"].values,
    size=len(df_defects),
    p=corridor_weights.values
)

# Merge in corridor metadata
df_defects = df_defects.merge(
    corridors[["corridor_id", "corridor_name", "length_km", "criticality"]],
    on="corridor_id",
    how="left"
)

# ------------------------------------------------------------
# 4. Build the unified maintenance task fields
# ------------------------------------------------------------
print("\n[5/5] Engineering realistic task attributes...")

def map_severity(s):
    """Map Kaggle's text severity to 1-5 numeric."""
    mapping = {"Low": 2, "Medium": 3, "High": 4, "Critical": 5}
    if pd.isna(s):
        return np.random.randint(1, 4)
    return mapping.get(str(s), 3)

def map_defect_type(failure_type, department):
    """Generate realistic defect description."""
    ft = str(failure_type).lower() if pd.notna(failure_type) else ""
    
    if department == "Engineering":
        options = ["Rail Fracture", "Track Geometry Deviation", "Loose Fastening",
                   "Ballast Deficiency", "Sleeper Crack", "Rail Wear"]
    elif department == "Signal & Telecom":
        options = ["Point Machine Failure", "Signal Relay Fault", "Track Circuit Failure",
                   "Axle Counter Fault", "Cable Cut", "Panel Indication Fault"]
    else:  # Traction
        options = ["OHE Wire Sag", "Insulator Flashover", "Bearing Overheating",
                   "Brake Pad Wear", "Transformer Oil Leak", "Pantograph Damage"]
    
    return np.random.choice(options)

# Apply
df_defects["severity"] = df_defects["failure_severity"].apply(map_severity)
df_defects["defect_type"] = df_defects.apply(
    lambda r: map_defect_type(r["failure_type"], r["department"]), axis=1
)

# Days overdue (based on last_maintenance_days)
df_defects["days_overdue"] = df_defects["last_maintenance_days"].fillna(0).astype(int).clip(0, 180)

# Estimated repair time (based on department and severity)
def estimate_repair_time(department, severity):
    base = {"Engineering": 60, "Signal & Telecom": 45, "Traction Distribution": 75}
    return int(base.get(department, 60) * (0.7 + 0.3 * severity) + np.random.randint(-15, 15))

df_defects["estimated_repair_time_min"] = df_defects.apply(
    lambda r: estimate_repair_time(r["department"], r["severity"]), axis=1
)

# Spatial position within corridor (random KM marker)
df_defects["spatial_start_km"] = np.random.uniform(0, df_defects["length_km"]).round(2)
df_defects["spatial_end_km"] = (df_defects["spatial_start_km"] + np.random.uniform(0.5, 2.0, len(df_defects))).round(2)

# Task ID
df_defects = df_defects.reset_index(drop=True)
df_defects["task_id"] = ["TASK_" + str(i+1).zfill(5) for i in range(len(df_defects))]

# ------------------------------------------------------------
# 5. Compute priority score
# ------------------------------------------------------------
def compute_priority(row, W1=0.5, W2=0.3, W3=0.2):
    s_norm = row["severity"] / 5.0
    c_norm = row["criticality"]  # already 0-1
    o_norm = min(1.0, row["days_overdue"] / 30.0)
    return round(W1 * s_norm + W2 * c_norm + W3 * o_norm, 3)

df_defects["priority_score"] = df_defects.apply(compute_priority, axis=1)

# ------------------------------------------------------------
# 6. Save the final maintenance tasks file
# ------------------------------------------------------------
final_cols = [
    "task_id", "department", "corridor_id", "corridor_name",
    "defect_type", "failure_type", "severity", "days_overdue",
    "estimated_repair_time_min", "spatial_start_km", "spatial_end_km",
    "priority_score", "criticality", "risk_score", "delay_minutes"
]

tasks = df_defects[final_cols].copy()
tasks.to_csv("data/processed/maintenance_tasks.csv", index=False)

print("\n" + "=" * 70)
print("MAINTENANCE TASKS BUILT")
print("=" * 70)
print(f"\n  Total tasks: {len(tasks)}")
print(f"  By department:\n{tasks['department'].value_counts().to_string()}")
print(f"\n  By severity:\n{tasks['severity'].value_counts().sort_index().to_string()}")
print(f"\n  Priority score stats:")
print(f"    Mean: {tasks['priority_score'].mean():.3f}")
print(f"    Max:  {tasks['priority_score'].max():.3f}")
print(f"    Min:  {tasks['priority_score'].min():.3f}")
print(f"\n  Sample (top 5 highest priority):")
print(tasks.nlargest(5, "priority_score")[
    ["task_id", "department", "corridor_name", "defect_type", "severity", "days_overdue", "priority_score"]
].to_string(index=False))

print("\n✓ Saved data/processed/maintenance_tasks.csv")