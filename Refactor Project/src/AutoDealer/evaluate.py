"""
evaluate.py

Runs a master-shaped CSV through the full preprocessing pipeline, feeds
the result into the already-trained model, and reports predicted price
next to actual price for every row.

Defaults to CONFIG.new_input_path - data the model has never seen in
training or testing. This is a genuinely honest performance check,
unlike scoring master.csv itself (which the model already trained on).
"""

import numpy as np
import pandas as pd

from .config import CONFIG
from .predict import predict
from .preprocess import build_tree_dataset
from .error_model import attach_shap_to_report


def score_master(master_path=CONFIG.new_input_path, model_path=CONFIG.model_path) -> pd.DataFrame:
    df = build_tree_dataset(master_path)

    predicted_price = predict(df, model_path=model_path)

    display_cols = [
        "Make", "Model", "Age", "mileage",
        "Cylinders", "Body Type", "Brand_Segment", "std_tier",
        "transmission", "Is_Hybrid",
        "no_accidents", "has_carfax", "one_owner", "service_records", "certified",
        CONFIG.target,
    ]
    report = df[[c for c in display_cols if c in df.columns]].copy()
    report = report.rename(columns={CONFIG.target: "actual_price"})
    report["predicted_price"] = predicted_price
    report["error"] = report["actual_price"] - report["predicted_price"]
    report["abs_error"] = report["error"].abs()
    report["pct_error"] = (report["error"] / report["actual_price"] * 100).round(2)
    report["unexplained_gap"] = report["error"] - report["expected_error"]
    # SHAP breakdown: for each row, how much each input is EXPECTED to
    # contribute to the price model's error for cars shaped like this one
    # (trained on the price model's own held-out residuals, see
    # error_model.py). expected_error is the single "how much should I
    # trust this prediction" number; the shap_<feature> columns are the
    # breakdown of which inputs are driving it.
    report = attach_shap_to_report(df, report)

    return report


if __name__ == "__main__":
    report = score_master()

    print(report.head(20).to_string(index=False))
    print()
    print(f"Rows scored: {len(report)}")
    print(f"MAE:  {report['abs_error'].mean():.2f}")
    print(f"RMSE: {np.sqrt((report['error'] ** 2).mean()):.2f}")

    output_path = CONFIG.model_path.parent / "predictions_vs_actual.csv"
    report.to_csv(output_path, index=False)
    print(f"\nFull comparison saved to {output_path}")