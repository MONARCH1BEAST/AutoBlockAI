"""
src/build_availability.py
Stage 3: Build corridor availability windows from RSTGCN data.

Input:
  - data/raw/rstgcn/train_routes_Sep2024.csv
  - data/raw/rstgcn/train_routes_delays_Sep2024.csv
  - data/processed/corridor_master.csv

Output:
  - data/processed/corridor_availability.csv
  - data/processed/train_delay_stats.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re

np.random.seed(42)

print("=" * 70)
print("STAGE 3: BUILDING CORRIDOR AVAILABILITY")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
print("\n[1/5] Loading RSTGCN train routes...")
routes = pd.read_csv("data/raw/rstgcn/train_routes_Sep2024.csv")
print(f"  Loaded {len(routes)} route records")
print(f"  Unique trains: {routes['trainNumber'].nunique()}")

print("\n[2/5] Loading RSTGCN delay data...")
delays = pd.read_csv("data/raw/rstgcn/train_routes_delays_Sep2024.csv", nrows=200000)
print(f"  Loaded {len(delays)} delay records (sample)")

print("\n[3/5] Loading corridor master...")
corridors = pd.read_csv("data/processed/corridor_master.csv")
print(f"  Loaded {len(corridors)} corridors")

# ------------------------------------------------------------
# 2. Clean time parsing helper
# ------------------------------------------------------------
def parse_time(t):
    """Parse 'HH:MM AM/PM' or 'HH:MM:SS' into minutes since midnight."""
    if pd.isna(t):
        return None
    t = str(t).strip().replace("'", "")
    try:
        # Handle "08:00 AM" format
        if "AM" in t.upper() or "PM" in t.upper():
            dt = datetime.strptime(t, "%I:%M %p")
        # Handle "08:00:00" format
        else:
            dt = datetime.strptime(t, "%H:%M:%S")
        return dt.hour * 60 + dt.minute
    except Exception:
        return None

# ------------------------------------------------------------
# 3. Build a "train journey" summary from routes
# ------------------------------------------------------------
print("\n[4/5] Building train journeys...")

# Aggregate per train: first departure, last arrival, total stops, total distance
routes["arr_min"] = routes["arrivalTime"].apply(parse_time)
routes["dep_min"] = routes["departureTime"].apply(parse_time)

journeys = routes.groupby("trainNumber").agg(
    train_name=("trainName", "first"),
    start_station=("station_code", "first"),
    end_station=("station_code", "last"),
    start_time_min=("dep_min", "first"),
    end_time_min=("arr_min", "last"),
    num_stops=("station_code", "count"),
    total_distance=("distance", "max"),
).reset_index()

# Handle overnight trains (end_time < start_time means it crosses midnight)
journeys["duration_min"] = journeys.apply(
    lambda r: r["end_time_min"] - r["start_time_min"]
    if r["end_time_min"] > r["start_time_min"]
    else r["end_time_min"] + 1440 - r["start_time_min"],
    axis=1
)

# Filter to trains with valid durations
journeys = journeys[
    (journeys["duration_min"].between(60, 1440)) &
    (journeys["start_time_min"].notna()) &
    (journeys["total_distance"] > 50)
].copy()

print(f"  Valid journeys: {len(journeys)}")

# ------------------------------------------------------------
# 4. Sample trains per corridor & build occupancy
# ------------------------------------------------------------
print("\n[5/5] Computing availability windows per corridor...")

availability_windows = []
delay_stats = []

for _, corridor in corridors.iterrows():
    corridor_id = corridor["corridor_id"]
    corridor_name = corridor["corridor_name"]
    criticality = corridor["criticality"]
    avg_daily = corridor["avg_daily_trains"]

    # Sample trains proportional to daily train count
    n_trains = min(int(avg_daily / 5), 30)  # Scale down for simulation
    sampled = journeys.sample(n=n_trains, random_state=int(criticality * 1000))

    # Build a 24-hour occupancy bitmap (1-min resolution)
    occupancy = np.zeros(1440, dtype=int)

    for _, train in sampled.iterrows():
        start = int(train["start_time_min"]) % 1440
        duration = int(train["duration_min"])
        for i in range(duration):
            occupancy[(start + i) % 1440] += 1

    # Find free windows (contiguous slots with 0 occupancy)
    free_windows = []
    in_window = False
    window_start = 0

    for minute in range(1440):
        if occupancy[minute] == 0 and not in_window:
            in_window = True
            window_start = minute
        elif occupancy[minute] > 0 and in_window:
            in_window = False
            window_length = minute - window_start
            if window_length >= 60:  # Only keep windows ≥ 60 min
                free_windows.append((window_start, minute, window_length))
    # Handle wrap-around at midnight
    if in_window:
        free_windows.append((window_start, 1440, 1440 - window_start))

    # If no large windows found, force-create 2 windows
    if not free_windows:
        free_windows = [
            (120, 300, 180),   # 02:00 - 05:00
            (780, 900, 120),   # 13:00 - 15:00
        ]

    # Merge into the availability table
    for start, end, length in free_windows:
        availability_windows.append({
            "corridor_id": corridor_id,
            "corridor_name": corridor_name,
            "window_start_min": start,
            "window_end_min": end,
            "window_length_min": length,
            "num_trains_sampled": n_trains,
            "criticality": criticality,
        })

    # Compute average delay for this corridor's sampled trains
    sampled_ids = sampled["trainNumber"].astype(str).tolist()
    corridor_delays = delays[delays["train"].astype(str).isin(sampled_ids)]
    if len(corridor_delays) > 0:
        avg_delay = corridor_delays["arr_delay"].mean()
        std_delay = corridor_delays["arr_delay"].std()
    else:
        avg_delay, std_delay = 0.0, 15.0

    delay_stats.append({
        "corridor_id": corridor_id,
        "corridor_name": corridor_name,
        "avg_delay_min": round(float(avg_delay) if not pd.isna(avg_delay) else 0.0, 2),
        "std_delay_min": round(float(std_delay) if not pd.isna(std_delay) else 15.0, 2),
        "sampled_trains": n_trains,
    })

# ------------------------------------------------------------
# 6. Save
# ------------------------------------------------------------
availability_df = pd.DataFrame(availability_windows)
delay_stats_df = pd.DataFrame(delay_stats)

availability_df.to_csv("data/processed/corridor_availability.csv", index=False)
delay_stats_df.to_csv("data/processed/train_delay_stats.csv", index=False)

print("\n" + "=" * 70)
print("CORRIDOR AVAILABILITY BUILT")
print("=" * 70)
print(f"\n  Total free windows: {len(availability_df)}")
print(f"  Corridors covered: {availability_df['corridor_id'].nunique()}")
print(f"\n  Window length stats (minutes):")
print(f"    Mean: {availability_df['window_length_min'].mean():.0f}")
print(f"    Max:  {availability_df['window_length_min'].max()}")
print(f"    Min:  {availability_df['window_length_min'].min()}")

print(f"\n  Sample windows for CORR_007 (Howrah-Chennai):")
print(availability_df[availability_df["corridor_id"] == "CORR_007"][
    ["window_start_min", "window_end_min", "window_length_min"]
].head(10).to_string(index=False))

print(f"\n  Delay statistics per corridor (top 5):")
print(delay_stats_df.nlargest(5, "avg_delay_min").to_string(index=False))

print("\n✓ Saved data/processed/corridor_availability.csv")
print("✓ Saved data/processed/train_delay_stats.csv")