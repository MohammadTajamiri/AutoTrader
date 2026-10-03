"""
error_model.py

Trains a SECOND model whose target isn't price - it's the price
model's own prediction error (actual - predicted) on the held-out
test set. SHAP values from this error model decompose, for any one
car, how much each input is expected to contribute to the price
model's error - and those contributions are GUARANTEED to sum exactly
to the error model's total predicted error for that car (the
"efficiency" property of Shapley values - see the additivity check at
the bottom of this file's test).

This answers a different question than the price model itself:
    price model        -> "what should this car cost?"
    error model + SHAP  -> "how much should I trust that number, and
                            which specific inputs are driving the
                            uncertainty or bias?"

Trained on SIGNED error (actual - predicted), not abs_error, so it
also tells you DIRECTION: a positive expected_error means the price
model typically UNDER-prices cars shaped like this one - which is
exactly the signal for "this one's worth a second look."
"""
import numpy as np
import pandas as pd
import joblib
import shap

from .config import CONFIG
from .data import load_and_split
from .pipeline import build_pipeline
from .predict import load_model as load_price_model


def build_error_training_data():
    """
    Reuses the price model's own train/test split, scores X_test with
    the ALREADY-TRAINED price model, and returns (X_test, signed_error)
    where signed_error = y_test - price_model_predictions.
    """
    X_train, X_test, y_train, y_test = load_and_split()
    price_model = load_price_model()
    y_pred = price_model.predict(X_test)
    signed_error = y_test.to_numpy() - y_pred
    return X_test, signed_error


def train_error_model():
    """
    Trains a second pipeline - same ColumnTransformer/OneHotEncoder/
    XGBRegressor shape as the price model (reuses build_pipeline(), so
    the two models are never accidentally out of sync on encoding) -
    but its target is the price model's own residual, not price.
    """
    X_test, signed_error = build_error_training_data()
    error_pipeline = build_pipeline()
    error_pipeline.fit(X_test, signed_error)

    CONFIG.error_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(error_pipeline, CONFIG.error_model_path)
    print(f"Trained error model on {len(X_test)} held-out rows, saved to {CONFIG.error_model_path}")
    return error_pipeline


def load_error_model(path=CONFIG.error_model_path):
    if not path.exists():
        raise FileNotFoundError(
            f"Error model not found: {path}. Run `python -m AutoDealer.error_model` first."
        )
    return joblib.load(path)


def _group_shap_by_original_feature(shap_row, feature_names):
    """
    OneHotEncoder expands one categorical column (e.g. Make) into many
    dummy columns (categorical__Make_BMW, categorical__Make_Audi, ...).
    TreeExplainer returns one SHAP value per DUMMY column, not per
    original feature. Since a one-hot-encoded row has exactly one "1"
    per original categorical feature and "0" everywhere else in that
    family, summing all of a family's SHAP values gives the correct
    total contribution of that ORIGINAL feature - verified against
    XGBRegressor's own raw prediction, see tests/test_error_model.py.

    Numeric/passthrough features (mileage, Age, ...) pass through the
    ColumnTransformer unchanged as a single column each, so they're
    just used directly under their own name.
    """
    totals = {}
    for name, value in zip(feature_names, shap_row):
        clean = name.split("__", 1)[-1]  # strip "categorical__"/"numeric__" prefix
        if clean in CONFIG.categorical_features:
            original = clean
        else:
            original = next(
                (c for c in CONFIG.categorical_features if clean.startswith(c + "_")),
                clean,
            )
        totals[original] = totals.get(original, 0.0) + value
    return totals


def explain_row(row_df: pd.DataFrame) -> dict:
    """
    Decomposes the error model's prediction for ONE car into a
    per-input dollar contribution, grouped back to your original
    feature names (not one-hot dummy columns).

    Returns {"base_value": ..., "expected_error": ...,
    "<feature>": shap_contribution, ...}. By construction:
        base_value + sum(shap contributions) == expected_error
    """
    error_pipeline = load_error_model()
    preprocessor = error_pipeline.steps[0][1]
    model = error_pipeline.steps[-1][1]

    X_transformed = preprocessor.transform(row_df)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed)

    grouped = _group_shap_by_original_feature(shap_values.values[0], feature_names)
    base_value = float(shap_values.base_values[0])
    expected_error = base_value + sum(grouped.values())

    return {"base_value": base_value, "expected_error": expected_error, **grouped}


def attach_shap_to_report(X: pd.DataFrame, report_df: pd.DataFrame) -> pd.DataFrame:
    """
    X is the full preprocessed feature dataframe already in the exact
    shape the trained pipelines expect - e.g. the `df` evaluate.py
    builds via build_tree_dataset(), the same object handed to
    predict(). report_df is the (possibly narrower, display-only)
    dataframe to attach the SHAP columns to - same row order/index as X.

    Adds one shap_<feature> column per original input, plus
    shap_base_value and expected_error. By construction, for every row:
        shap_base_value + sum(shap_<feature> columns) == expected_error
    """
    error_pipeline = load_error_model()
    preprocessor = error_pipeline.steps[0][1]
    model = error_pipeline.steps[-1][1]

    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed)

    grouped_rows = [
        _group_shap_by_original_feature(row, feature_names)
        for row in shap_values.values
    ]
    grouped_df = pd.DataFrame(grouped_rows, index=X.index).fillna(0.0)
    grouped_df.columns = [f"shap_{c}" for c in grouped_df.columns]

    report_df = report_df.copy()
    report_df["shap_base_value"] = shap_values.base_values
    report_df = pd.concat([report_df, grouped_df], axis=1)
    report_df["expected_error"] = report_df["shap_base_value"] + grouped_df.sum(axis=1)

    return report_df


if __name__ == "__main__":
    train_error_model()