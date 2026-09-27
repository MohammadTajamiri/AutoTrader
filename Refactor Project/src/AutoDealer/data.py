"""
data.py

Loading and validation for the modeling dataset. This is the seam that
will change when Phase 3 swaps the raw input from TreeInput.csv to
master.csv (plus the full preprocessing pipeline) — everything downstream
(features.py, pipeline.py, train.py) depends only on this module's
public functions, not on how the raw file is produced.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CONFIG


def load_raw_data() -> pd.DataFrame:
    """Loads the raw modeling dataset from CONFIG.raw_data_path."""
    if not CONFIG.raw_data_path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {CONFIG.raw_data_path}. "
            f"Check that the file exists and CONFIG.raw_data_path is correct."
        )
    return pd.read_csv(CONFIG.raw_data_path)


def validate_columns(df: pd.DataFrame) -> None:
    """
    Confirms every column CONFIG says it needs (target + numeric +
    categorical features) actually exists in df. Fails loudly and
    specifically rather than letting a KeyError surface later, deep
    inside training.
    """
    expected = {CONFIG.target, *CONFIG.numeric_features, *CONFIG.categorical_features}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns in raw data: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


def get_train_test_split(df: pd.DataFrame):
    """
    Splits df into train/test sets using CONFIG.test_size and
    CONFIG.random_state, stratified on Make (to avoid the
    under-representation problem rare makes hit under plain random
    splitting).

    Returns
    -------
    X_train, X_test, y_train, y_test : pandas objects
    """
    feature_cols = list(CONFIG.numeric_features) + list(CONFIG.categorical_features)

    X = df[feature_cols]
    y = df[CONFIG.target]

    stratify_col = df["Make"] if "Make" in df.columns else None

    return train_test_split(
        X, y,
        test_size=CONFIG.test_size,
        random_state=CONFIG.random_state,
        stratify=stratify_col,
    )


def load_and_split():
    """Convenience wrapper: load, validate, and split in one call."""
    df = load_raw_data()
    validate_columns(df)
    return get_train_test_split(df)