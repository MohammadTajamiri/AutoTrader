"""
predict.py

Loads the saved pipeline artifact and predicts prices for new data.
"""

from pathlib import Path

import joblib
import pandas as pd

from .config import CONFIG
from .data import validate_columns


def load_model(model_path=CONFIG.model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Run `python -m AutoDealer.train` first to create it."
        )
    return joblib.load(model_path)


def predict(df: pd.DataFrame, model_path=CONFIG.model_path):
    model = load_model(model_path)
    validate_columns(df, require_target=False)

    feature_cols = list(CONFIG.numeric_features) + list(CONFIG.categorical_features)
    return model.predict(df[feature_cols])