"""
src/utils/fix_criticality.py
Fixes the criticality scores in corridor_master.csv 
using real-world Indian Railways knowledge.
"""

import pandas as pd

print("=" * 70)
print("FIXING CORRIDOR CRITICALITY SCORES")
print("=" * 70)

# Load existing corridor master
df = pd.read_csv("data/processed/corridor_master.csv")
print(f"\nLoaded {len(df)} corridors")

# ------------------------------------------------------------
# Rule 1: Known high-importance corridors (Tier 1 - Golden Quadrilateral + DFCs)
# ------------------------------------------------------------
TIER_1 = [
    "Howrah - Delhi Railway",
    "Mumbai - Delhi Railway",
    "Howrah - Chennai Main Line",
    "Mumbai-Pune Railway",
    "Western Dedicated Freight Corridor",
    "Eastern Dedicated Freight Corridor",
    "Delhi - Kalka Line",
    "Lucknow–Moradabad line",
]

# ------------------------------------------------------------
# Rule 2: Medium-importance corridors
# ------------------------------------------------------------
TIER_2 = [
    "Konkan Railway",
    "Bina - Katni railway",
    "Katni - Obra railway",
    "Mumbai - Delhi Railway",
    "Kolkata Circular Railway",
    "Bina - Katni railway",
]

# ------------------------------------------------------------
# Rule 3: Compute criticality using multiple factors
# ------------------------------------------------------------
def compute_criticality(row):
    """
    Formula:
    criticality = 0.5 * tier_score + 0.3 * length_score + 0.2 * density_score
    """
    # Tier score (0.4 to 1.0)
    if row["corridor_name"] in TIER_1:
        tier = 1.0
    elif row["corridor_name"] in TIER_2:
        tier = 0.7
    else:
        tier = 0.5

    # Length score (longer corridors carry more traffic)
    # Normalize: 100km = 0.3, 3500km = 1.0
    length_score = min(1.0, 0.3 + (row["length_km"] / 3500) * 0.7)

    # Segment density score (more segments = more infrastructure = more important)
    seg_score = min(1.0, row["num_segments"] / 1500)

    return round(0.5 * tier + 0.3 * length_score + 0.2 * seg_score, 2)


df["criticality"] = df.apply(compute_criticality, axis=1)

# ------------------------------------------------------------
# Rule 4: Assign realistic daily train counts
# ------------------------------------------------------------
def compute_train_count(row):
    """
    Realistic train density based on corridor type.
    """
    if "Dedicated Freight Corridor" in row["corridor_name"]:
        return 180  # DFCs are designed for high capacity
    elif row["corridor_name"] in TIER_1:
        return 150  # Golden Quadrilateral routes
    elif row["corridor_name"] in TIER_2:
        return 90   # Medium density
    else:
        return 40   # Regional / feeder routes


df["avg_daily_trains"] = df.apply(compute_train_count, axis=1)

# ------------------------------------------------------------
# Save the corrected file
# ------------------------------------------------------------
df.to_csv("data/processed/corridor_master.csv", index=False)

print("\n" + "=" * 70)
print("CORRECTED CORRIDOR MASTER")
print("=" * 70)
print(df[["corridor_id", "corridor_name", "length_km", "avg_daily_trains", "criticality"]].to_string(index=False))

print("\n✓ Saved data/processed/corridor_master.csv")