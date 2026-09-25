"""
src/build_corridors.py
Stage 1: Build the Corridor Master file.

Input:
  - data/raw/osm_railways/railways_lines.shp
  - data/raw/rstgcn/IRN_edges.csv

Output:
  - data/processed/corridor_master.csv
"""

import pandas as pd
import geopandas as gpd
from shapely.ops import linemerge
import os

print("=" * 70)
print("STAGE 1: BUILDING CORRIDOR MASTER")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load OSM Railways
# ------------------------------------------------------------
print("\n[1/4] Loading OSM Railways shapefile...")
osm = gpd.read_file("data/raw/osm_railways/railways_lines.shp")
print(f"  Loaded {len(osm)} line segments")

# ------------------------------------------------------------
# 2. Filter to named major railway lines
# ------------------------------------------------------------
print("\n[2/4] Filtering to named railway corridors...")
# Keep only segments that have a name (these are typically major routes)
osm_named = osm[osm["name"].notna()].copy()
print(f"  Named segments: {len(osm_named)}")

# Show the top named railways
top_names = osm_named["name"].value_counts().head(20)
print(f"\n  Top 20 named railways:")
for name, count in top_names.items():
    print(f"    {name}: {count} segments")

# ------------------------------------------------------------
# 3. Pick 20 major corridors (customize this list!)
# ------------------------------------------------------------
print("\n[3/4] Selecting 20 major corridors for the project...")

# You can manually pick the most important ones, or auto-pick top 20
selected_names = top_names.head(20).index.tolist()

# Filter to only those
osm_corridors = osm_named[osm_named["name"].isin(selected_names)].copy()

# Create a corridor_id for each named railway
corridor_master = (
    osm_corridors.groupby("name")
    .agg(
        num_segments=("geometry", "count"),
        geometry=("geometry", lambda x: linemerge(list(x))),
        region=("adm1_name", lambda x: x.mode().iloc[0] if not x.mode().empty else None),
        districts=("adm2_name", lambda x: ", ".join(x.dropna().unique()[:3])),
    )
    .reset_index()
    .rename(columns={"name": "corridor_name"})
)

# Add a synthetic corridor_id
corridor_master["corridor_id"] = ["CORR_" + str(i).zfill(3) for i in range(1, len(corridor_master) + 1)]

# Compute length in km (project to a metric CRS)
corridor_master_gdf = gpd.GeoDataFrame(corridor_master, geometry="geometry", crs=osm.crs)
corridor_master_gdf = corridor_master_gdf.to_crs(epsg=3857)  # Web Mercator for length calc
corridor_master["length_km"] = (corridor_master_gdf.geometry.length / 1000).round(2)

print(f"\n  Selected {len(corridor_master)} corridors")
print(corridor_master[["corridor_id", "corridor_name", "region", "num_segments", "length_km"]].to_string(index=False))

# ------------------------------------------------------------
# 4. Attach train frequency from RSTGCN Edges
# ------------------------------------------------------------
print("\n[4/4] Attaching train frequency from RSTGCN edges...")

# Load RSTGCN edges
edges = pd.read_csv("data/raw/rstgcn/IRN_edges.csv")

# Compute average trains per corridor region (approximate mapping)
# We'll use total trains in the network as a density proxy
avg_trains = edges["ntrains"].mean()
max_trains = edges["ntrains"].max()

# Assign a synthetic train density to each corridor
# (in a full version, you'd map OSM corridors to RSTGCN edges via geometry)
import numpy as np
np.random.seed(42)
corridor_master["avg_daily_trains"] = np.random.randint(
    int(avg_trains * 0.5), int(max_trains * 0.8), size=len(corridor_master)
).tolist()

# Add a criticality score (0 to 1) based on train density
corridor_master["criticality"] = (
    corridor_master["avg_daily_trains"] / corridor_master["avg_daily_trains"].max()
).round(2)

# ------------------------------------------------------------
# 5. Save
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("SAVING CORRIDOR MASTER")
print("=" * 70)

os.makedirs("data/processed", exist_ok=True)

# Save only the CSV (drop geometry for now)
corridor_master.drop(columns=["geometry"]).to_csv(
    "data/processed/corridor_master.csv", index=False
)

# Also save the GeoJSON for map visualization later
corridor_master_gdf_wgs = corridor_master_gdf.to_crs(epsg=4326)
corridor_master_gdf_wgs.to_file(
    "data/processed/corridor_master.geojson", driver="GeoJSON"
)

print(f"\n  ✓ Saved data/processed/corridor_master.csv ({len(corridor_master)} corridors)")
print(f"  ✓ Saved data/processed/corridor_master.geojson (for map visualization)")
print(f"\n  Preview:")
print(corridor_master[["corridor_id", "corridor_name", "region", "length_km", "avg_daily_trains", "criticality"]].to_string(index=False))