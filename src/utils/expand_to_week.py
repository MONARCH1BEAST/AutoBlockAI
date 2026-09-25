"""
src/utils/expand_to_week.py
Expand corridor availability to a 7-day horizon.
"""

import pandas as pd

print("Expanding availability to 7-day horizon...")

df = pd.read_csv("data/processed/corridor_availability.csv")

# Replicate each window across 7 days
expanded = []
for day in range(7):
    day_df = df.copy()
    day_df["day_of_week"] = day  # 0=Monday, 6=Sunday
    day_df["window_id"] = (
        day_df["corridor_id"] + "_D" + str(day) + "_W" +
        day_df.groupby("corridor_id").cumcount().astype(str)
    )
    expanded.append(day_df)

result = pd.concat(expanded, ignore_index=True)
result.to_csv("data/processed/corridor_availability_week.csv", index=False)

print(f"  Original windows: {len(df)}")
print(f"  Expanded windows: {len(result)}")
print(f"  Saved: data/processed/corridor_availability_week.csv")