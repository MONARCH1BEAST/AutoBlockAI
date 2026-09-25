"""
src/models/golden_block_optimizer.py
STAGE 4: The Golden Block Optimizer (v3 - Per-Corridor).

Architectural change:
  Runs the optimizer INDEPENDENTLY per corridor, mirroring how
  real railway Divisions plan their maintenance blocks.
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

np.random.seed(42)

print("=" * 70)
print("STAGE 4: GOLDEN BLOCK OPTIMIZER (v3 - Per-Corridor)")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/6] Loading data...")
tasks = pd.read_csv("data/processed/maintenance_tasks.csv")
windows = pd.read_csv("data/processed/corridor_availability_week.csv")
delays = pd.read_csv("data/processed/train_delay_stats.csv")
corridors = pd.read_csv("data/processed/corridor_master.csv")

print(f"  Total tasks: {len(tasks)}")
print(f"  Total windows: {len(windows)}")
print(f"  Total corridors: {len(corridors)}")

# ============================================================
# 2. CLEAN DELAY STATS
# ============================================================
print("\n[2/6] Cleaning delay stats...")
delays["std_delay_min"] = delays["std_delay_min"].clip(upper=60)
delays["avg_delay_min"] = delays["avg_delay_min"].clip(upper=90)
delay_lookup = delays.set_index("corridor_id")[["avg_delay_min", "std_delay_min"]].to_dict("index")
print(f"  Delay stats cleaned and indexed by corridor")

# ============================================================
# 3. PER-CORRIDOR PROCESSING
# ============================================================
print("\n[3/6] Running per-corridor optimization...")

TASKS_PER_CORRIDOR = 20  # Top N tasks per corridor
all_assignments = []
corridor_summary = []

for _, corridor in corridors.iterrows():
    corridor_id = corridor["corridor_id"]
    corridor_name = corridor["corridor_name"]

    # --- Get tasks for this corridor only ---
    corridor_tasks = tasks[tasks["corridor_id"] == corridor_id].copy()
    if len(corridor_tasks) == 0:
        continue

    # Take top N tasks by priority
    corridor_tasks = corridor_tasks.nlargest(TASKS_PER_CORRIDOR, "priority_score").reset_index(drop=True)

    # --- Get windows for this corridor only ---
    corridor_windows = windows[windows["corridor_id"] == corridor_id].copy()
    if len(corridor_windows) == 0:
        continue

    # --- Greedy assignment per window (parallel execution) ---
    remaining_tasks = corridor_tasks.to_dict("records")
    assigned_this_corridor = 0

    for w_idx, window in corridor_windows.iterrows():
        window_length = window["window_length_min"]

        # Pick 1 task per department (Golden Block forcing)
        depts_picked = set()
        tasks_this_window = []
        remaining_after = []

        for task in remaining_tasks:
            dept = task["department"]
            if dept not in depts_picked and task["estimated_repair_time_min"] <= window_length:
                tasks_this_window.append(task)
                depts_picked.add(dept)
            else:
                remaining_after.append(task)

        # Pass 2: Fill remaining slots (up to 6 tasks per window)
        still_remaining = []
        for task in remaining_after:
            if len(tasks_this_window) < 6 and task["estimated_repair_time_min"] <= window_length:
                tasks_this_window.append(task)
            else:
                still_remaining.append(task)

        remaining_tasks = still_remaining

        # Record
        for task in tasks_this_window:
            all_assignments.append({
                "task_id": task["task_id"],
                "window_id": window["window_id"],
                "corridor_id": corridor_id,
                "corridor_name": corridor_name,
                "department": task["department"],
                "defect_type": task["defect_type"],
                "severity": task["severity"],
                "estimated_repair_time_min": task["estimated_repair_time_min"],
                "priority_score": task["priority_score"],
                "day_of_week": window["day_of_week"],
                "window_start_min": window["window_start_min"],
                "window_end_min": window["window_end_min"],
                "window_length_min": window_length,
            })
            assigned_this_corridor += 1

    corridor_summary.append({
        "corridor_id": corridor_id,
        "corridor_name": corridor_name,
        "tasks_available": len(corridor_tasks),
        "tasks_assigned": assigned_this_corridor,
    })

schedule_df = pd.DataFrame(all_assignments)
print(f"  Total assignments: {len(schedule_df)}")
print(f"  Corridors covered: {schedule_df['corridor_id'].nunique()}")

# ============================================================
# 4. DETECT GOLDEN BLOCKS
# ============================================================
print("\n[4/6] Detecting Golden Blocks...")

if len(schedule_df) == 0:
    print("  ERROR: No tasks assigned. Aborting.")
    exit()

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
        total_repair_min=("estimated_repair_time_min", "sum"),
        max_task_duration=("estimated_repair_time_min", "max"),
        avg_priority=("priority_score", "mean"),
    )
    .reset_index()
)

golden_blocks["is_golden"] = golden_blocks["num_departments"] >= 2
golden_blocks = golden_blocks.sort_values(
    ["is_golden", "num_tasks"], ascending=[False, False]
).reset_index(drop=True)

# ============================================================
# 5. MONTE CARLO REGRET SIMULATION
# ============================================================
print("\n[5/6] Running Monte Carlo Regret Simulation...")

def compute_clash_probability(block_duration, avg_delay, std_delay):
    if std_delay <= 0:
        return 0.0
    z = (block_duration - avg_delay) / std_delay
    return round(float(1 - scipy_stats.norm.cdf(z)), 3)

for idx, block in golden_blocks.iterrows():
    stats = delay_lookup.get(block["corridor_id"], {"avg_delay_min": 20, "std_delay_min": 15})
    clash_prob = compute_clash_probability(
        block["window_length_min"], stats["avg_delay_min"], stats["std_delay_min"]
    )
    golden_blocks.at[idx, "clash_probability"] = clash_prob
    golden_blocks.at[idx, "operationally_safe"] = clash_prob < 0.10

# ============================================================
# 6. SAVE OUTPUTS
# ============================================================
print("\n[6/6] Saving outputs...")
schedule_df.to_csv("output/schedules/scheduled_blocks.csv", index=False)
golden_blocks.to_csv("output/schedules/golden_blocks.csv", index=False)
pd.DataFrame(corridor_summary).to_csv("output/reports/corridor_summary.csv", index=False)

# ============================================================
# 7. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("GOLDEN BLOCK OPTIMIZATION COMPLETE (v3 - Per-Corridor)")
print("=" * 70)

num_golden = golden_blocks["is_golden"].sum()
num_full_golden = (golden_blocks["num_departments"] == 3).sum()
num_safe = golden_blocks["operationally_safe"].sum()
total_blocks = len(golden_blocks)
traditional = len(schedule_df)
reduction = (1 - total_blocks / traditional) * 100 if traditional > 0 else 0

print(f"\n📊 KEY METRICS:")
print(f"  Total tasks assigned: {len(schedule_df)}")
print(f"  Corridors covered: {schedule_df['corridor_id'].nunique()}/20")
print(f"  Total blocks used: {total_blocks}")
print(f"  Golden Blocks (2+ depts): {num_golden}")
print(f"  Full Golden (3 depts):    {num_full_golden}")
print(f"  Operationally safe: {num_safe}/{total_blocks} ({num_safe/total_blocks*100:.0f}%)")

print(f"\n🎯 EFFICIENCY GAIN:")
print(f"  Traditional (1 task = 1 block): {traditional} blocks")
print(f"  Golden Block approach:          {total_blocks} blocks")
print(f"  Block reduction:                {reduction:.1f}%")

print(f"\n📋 CORRIDOR-WISE BREAKDOWN:")
summary_df = pd.DataFrame(corridor_summary)
print(summary_df.sort_values("tasks_assigned", ascending=False).to_string(index=False))

print(f"\n🏆 TOP 5 GOLDEN BLOCKS:")
top_golden = golden_blocks[golden_blocks["is_golden"]].head(5)
for _, b in top_golden.iterrows():
    print(f"\n  {b['window_id']} | {b['corridor_name']}")
    print(f"    Tasks: {b['num_tasks']} | Departments: {b['departments']}")
    print(f"    Clash probability: {b['clash_probability']*100:.1f}%")

print(f"\n  ✓ Saved output/schedules/scheduled_blocks.csv")
print(f"  ✓ Saved output/schedules/golden_blocks.csv")
print(f"  ✓ Saved output/reports/corridor_summary.csv")