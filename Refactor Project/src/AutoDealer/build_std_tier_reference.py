"""
build_std_tier_reference.py

Builds the std_tier reference table from a fuller dataset (not
necessarily the same master.csv used for training), applying the same
cleaning rules as the Used-car pipeline: year filtering, and the
Make/Cylinders exclusions. The std_tier itself is still computed ONLY
from Condition == "New" rows within that dataset - leak-free, since New
rows are a structurally separate population from whatever Used rows the
lookup eventually gets merged onto in preprocess.py.

Usage:
    python -m AutoDealer.build_std_tier_reference [path_to_source_csv]

    If no path is given, defaults to CONFIG.std_tier_source_path.
"""

import sys

import pandas as pd

from .config import CONFIG
from .preprocess import apply_exclusions, clean_master_columns, filter_by_year, load_master, merge_engine_specs


def build_std_tier_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a (Make, Model) -> std_tier table using ONLY rows where
    Condition == "New". Also records how many New listings each tier
    was computed from, for transparency - a tier based on 1-2 listings
    is much less trustworthy than one based on 50.
    """
    new_cars = df[df["Condition"] == "New"]

    rows = []
    for make, group in new_cars.groupby("Make"):
        make_mean = group["price"].mean()
        make_std = group["price"].std()
        model_counts = group.groupby("Model").size()
        model_avgs = group.groupby("Model")["price"].mean()

        for model, avg_price in model_avgs.items():
            if avg_price > make_mean + 3 * make_std:
                tier = "3std"
            elif avg_price > make_mean + 2 * make_std:
                tier = "2std"
            elif avg_price > make_mean + 1 * make_std:
                tier = "1std"
            else:
                tier = "Normal"
            rows.append({
                "Make": make,
                "Model": model,
                "std_tier": tier,
                "new_listing_count": int(model_counts[model]),
                "avg_new_price": round(avg_price, 2),
            })

    return pd.DataFrame(rows, columns=["Make", "Model", "std_tier", "new_listing_count", "avg_new_price"])


def build_and_save(source_path=CONFIG.std_tier_source_path):
    df = load_master(source_path)
    df = clean_master_columns(df)
    df = filter_by_year(df, min_year=CONFIG.min_manufacture_year)
    df = merge_engine_specs(df)   # needed so Cylinders exists, for the Electric exclusion below
    df = apply_exclusions(df)     # drops Land Rover, Porsche, Electric - same rules as training data

    lookup = build_std_tier_lookup(df)

    CONFIG.std_tier_reference_path.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(CONFIG.std_tier_reference_path, index=False)

    print(f"Source: {source_path}")
    print(f"New-condition listings used: {len(df[df['Condition'] == 'New'])}")
    print(f"Covers {len(lookup)} (Make, Model) combinations.\n")

    print("Tier counts overall:")
    print(lookup["std_tier"].value_counts(), "\n")

    print("Tier distribution by Make:")
    print(lookup.groupby(["Make", "std_tier"]).size().unstack(fill_value=0), "\n")

    print("Full lookup (sorted by Make):")
    print(lookup.sort_values(["Make", "std_tier"]).to_string(index=False))

    print(f"\nSaved to {CONFIG.std_tier_reference_path}")

    return lookup


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else CONFIG.std_tier_source_path
    build_and_save(source)