"""
inference.py

Turns ONE manually-entered car (as typed into the Streamlit dashboard)
into the same feature shape build_tree_dataset() produces for training
and evaluation - WITHOUT the batch-only steps that only make sense
across many rows (price-bin filtering, rare-Make dropping, high-NaN
column dropping). A single new car has no price yet and is, by
definition, a sample of one, so those steps don't apply - running it
through build_tree_dataset() itself would crash (no price column) or
get the row dropped outright.

What IS still needed, because it's per-row feature engineering rather
than batch filtering: the enrichment lookup (for Brand_Segment), the
engine-spec merge (for Cylinders/Body Type), the std_tier reference
merge, and the Age calculation - the same four lookups a training row
goes through, just run on a dataframe of one.
"""
import pandas as pd

from .config import CONFIG
from .enrichment import enrich_dataframe
from .preprocess import merge_engine_specs, apply_std_tier, compute_age


def preprocess_single_input(raw: dict) -> pd.DataFrame:
    """
    raw must contain: Make, Model, trim, Year, mileage, transmission,
    fuel_type, Is_Hybrid, no_accidents, has_carfax, one_owner,
    service_records, certified - the same car-level facts a dashboard
    user types in or picks from a dropdown. (trim/fuel_type are kept
    even though Is_Hybrid is set directly, in case enrich_dataframe's
    lookups reference them.)

    Returns a one-row DataFrame with exactly CONFIG.numeric_features +
    CONFIG.categorical_features, ready to hand to predict() or
    error_model.explain_row().
    """
    df = pd.DataFrame([raw])

    df = enrich_dataframe(df)          # -> Brand_Segment
    df = merge_engine_specs(df)        # -> Cylinders, Body Type (or "Unknown" if no match)
    df = apply_std_tier(df)            # -> std_tier (or "Normal" if no match)
    df = compute_age(df, current_year=CONFIG.current_year)  # -> Age

    feature_cols = list(CONFIG.numeric_features) + list(CONFIG.categorical_features)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"preprocess_single_input() is missing required feature columns {missing} "
            f"after running the enrichment/merge steps - check that `raw` included "
            f"everything those lookups need."
        )
    return df[feature_cols]