"""
src/models/feature_importance_plot.py

Generate a clean feature importance chart from the trained model.

Output:
  - output/reports/feature_importance.png
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

print("=" * 70)
print("FEATURE IMPORTANCE VISUALIZATION")
print("=" * 70)

# ============================================================
# 1. LOAD MODEL
# ============================================================
print("\n[1/2] Loading trained model...")

with open("src/models/saved/assignment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("src/models/saved/feature_columns.pkl", "rb") as f:
    feature_cols = pickle.load(f)

print(f"  Model loaded with {len(feature_cols)} features")

# ============================================================
# 2. BUILD IMPORTANCE DATAFRAME
# ============================================================
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True)

# Pretty labels
pretty_labels = {
    "window_has_capacity": "Window Fits Task",
    "same_corridor": "Correct Corridor",
    "tod_sin": "Time-of-Day (sin)",
    "tod_cos": "Time-of-Day (cos)",
    "day_of_week": "Day of Week",
    "severity_x_criticality": "Severity x Corridor Criticality",
    "window_length": "Window Length",
    "window_start_hour": "Window Start Hour",
    "repair_time_fit": "Repair Time Fit",
    "dept_Traction_Distribution": "Dept: Traction Distribution",
    "dept_Engineering": "Dept: Engineering",
    "dept_Signal_and_Telecom": "Dept: Signal & Telecom",
    "repair_time": "Repair Time",
    "priority_score": "Priority Score",
    "priority_x_criticality": "Priority x Criticality",
    "criticality": "Corridor Criticality",
    "corridor_criticality": "Corridor Criticality",
    "severity": "Severity",
    "days_overdue": "Days Overdue",
    "corridor_avg_delay": "Corridor Avg Delay",
    "corridor_std_delay": "Corridor Delay Std",
}

importance["pretty"] = importance["feature"].map(pretty_labels).fillna(importance["feature"])

# Take top 15
top_n = 15
top = importance.tail(top_n)

# ============================================================
# 3. PLOT
# ============================================================
print("\n[2/2] Generating chart...")

fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0e1117")
ax.set_facecolor("#0e1117")

# Color bars by category
colors = []
for f in top["feature"]:
    if "corridor" in f or "dept" in f:
        colors.append("#3b82f6")  # blue - context
    elif "tod" in f or "day" in f or "window_start" in f:
        colors.append("#facc15")  # yellow - temporal
    else:
        colors.append("#00d4ff")  # cyan - task

bars = ax.barh(top["pretty"], top["importance"], color=colors, edgecolor="none")

# Add value labels
for bar, val in zip(bars, top["importance"]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", color="#fafafa", fontsize=10)

# Style
ax.set_xlabel("Feature Importance", color="#fafafa", fontsize=12)
ax.set_title("XGBoost Feature Importance - Top 15",
             color="#00d4ff", fontsize=16, weight="bold", pad=20)
ax.tick_params(colors="#fafafa", labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#2d3654")
ax.spines["bottom"].set_color("#2d3654")
ax.grid(axis="x", color="#2d3654", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

# Legend
legend_elements = [
    Patch(facecolor="#00d4ff", label="Task Feature"),
    Patch(facecolor="#facc15", label="Temporal Feature"),
    Patch(facecolor="#3b82f6", label="Context Feature"),
]
ax.legend(handles=legend_elements, loc="lower right",
          facecolor="#1a1f2e", edgecolor="#2d3654",
          labelcolor="#fafafa", fontsize=10)

plt.tight_layout()
os.makedirs("output/reports", exist_ok=True)
plt.savefig("output/reports/feature_importance.png", dpi=150,
            facecolor="#0e1117", edgecolor="none")
plt.close()

print(f"  Saved output/reports/feature_importance.png")
print(f"\n  Top 5 features:")
for _, r in top.tail(5).iloc[::-1].iterrows():
    print(f"    {r['pretty']:40s} {r['importance']:.4f}")

print(f"\n{'=' * 70}")
print("FEATURE IMPORTANCE COMPLETE")
print("=" * 70)