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


def load_raw_data(path=CONFIG.raw_data_path) -> pd.DataFrame:
    """Loads the raw modeling dataset from CONFIG.raw_data_path."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}. "
            f"Check that the file exists and CONFIG.raw_data_path is correct."
        )
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, require_target: bool = True) -> None:
    """
    Confirms every column CONFIG needs actually exists in df.
    Set require_target=False when validating new data to predict on
    (new listings don't have a price yet).
    """
    expected = {*CONFIG.numeric_features, *CONFIG.categorical_features}
    if require_target:
        expected.add(CONFIG.target)

    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns: {sorted(missing)}. "
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