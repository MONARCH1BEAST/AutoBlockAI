"""
verify_data.py
Confirms all 4 datasets load correctly and prints their structure.
Run: python verify_data.py
"""

import pandas as pd
import json
import geopandas as gpd
import os

print("=" * 70)
print("AUTOBLOCK AI - DATA VERIFICATION")
print("=" * 70)

# ============================================================
# 1. RSTGCN DATASET
# ============================================================
print("\n[1/4] RSTGCN Train Delay Dataset")
print("-" * 70)

try:
    # 1a. Network edges
    edges = pd.read_csv("data/raw/rstgcn/IRN_edges.csv")
    print(f"\n  ✓ IRN_edges.csv")
    print(f"    Shape: {edges.shape}")
    print(f"    Columns: {edges.columns.tolist()}")
    print(f"    Sample:\n{edges.head(3).to_string()}")

    # 1b. Station-zone mapping
    with open("data/raw/rstgcn/stations_zones_mapping.json") as f:
        zones = json.load(f)
    print(f"\n  ✓ stations_zones_mapping.json")
    print(f"    Total stations: {len(zones)}")
    print(f"    Sample: {list(zones.items())[:3]}")

    # 1c. Scheduled routes
    routes = pd.read_csv("data/raw/rstgcn/train_routes_Sep2024.csv")
    print(f"\n  ✓ train_routes_Sep2024.csv")
    print(f"    Shape: {routes.shape}")
    print(f"    Columns: {routes.columns.tolist()}")
    print(f"    Sample:\n{routes.head(3).to_string()}")

    # 1d. Flattened delays
    delays = pd.read_csv("data/raw/rstgcn/train_routes_delays_Sep2024.csv")
    print(f"\n  ✓ train_routes_delays_Sep2024.csv")
    print(f"    Shape: {delays.shape}")
    print(f"    Columns: {delays.columns.tolist()}")
    print(f"    Sample:\n{delays.head(3).to_string()}")

    # 1e. Raw delay JSON (just check it loads)
    with open("data/raw/rstgcn/train_delays_Sep2024.json") as f:
        raw_delays = json.load(f)
    print(f"\n  ✓ train_delays_Sep2024.json")
    print(f"    Total trains: {len(raw_delays)}")
    sample_train = list(raw_delays.keys())[:1]
    print(f"    Sample train ID: {sample_train}")

except Exception as e:
    print(f"  ✗ ERROR in RSTGCN: {e}")

# ============================================================
# 2. KAGGLE FAILURE DETECTION (100K)
# ============================================================
print("\n\n[2/4] Kaggle Failure Detection Dataset")
print("-" * 70)

try:
    folder = "data/raw/kaggle_failure"
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not csv_files:
        print("  ✗ No CSV file found in folder.")
    else:
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(folder, csv_file))
            print(f"\n  ✓ {csv_file}")
            print(f"    Shape: {df.shape}")
            print(f"    Columns: {df.columns.tolist()}")
            print(f"    Sample:\n{df.head(3).to_string()}")
            print(f"\n    Data types:\n{df.dtypes.to_string()}")

except Exception as e:
    print(f"  ✗ ERROR in Kaggle Failure: {e}")

# ============================================================
# 3. KAGGLE TIMETABLE
# ============================================================
print("\n\n[3/4] Kaggle Indian Railways Timetable")
print("-" * 70)

try:
    folder = "data/raw/kaggle_timetable"
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not csv_files:
        print("  ✗ No CSV file found in folder.")
    else:
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(folder, csv_file))
            print(f"\n  ✓ {csv_file}")
            print(f"    Shape: {df.shape}")
            print(f"    Columns: {df.columns.tolist()}")
            print(f"    Sample:\n{df.head(3).to_string()}")

except Exception as e:
    print(f"  ✗ ERROR in Kaggle Timetable: {e}")

# ============================================================
# 4. OSM RAILWAYS
# ============================================================
print("\n\n[4/4] OSM Railways Shapefile")
print("-" * 70)

try:
    shp_path = "data/raw/osm_railways/railways_lines.shp"
    gdf = gpd.read_file(shp_path)
    print(f"\n  ✓ railways_lines.shp")
    print(f"    Total line segments: {len(gdf)}")
    print(f"    Columns: {gdf.columns.tolist()}")
    print(f"    CRS: {gdf.crs}")
    print(f"    Sample:\n{gdf.head(3).to_string()}")

except Exception as e:
    print(f"  ✗ ERROR in OSM Railways: {e}")

# ============================================================
# SUMMARY
# ============================================================
print("\n\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)