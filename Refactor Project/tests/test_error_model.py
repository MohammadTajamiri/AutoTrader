import numpy as np
from AutoDealer.error_model import _group_shap_by_original_feature


def test_grouped_shap_sums_to_model_prediction():
    """
    The entire point of using SHAP is that per-feature contributions
    sum exactly to the model's own prediction (Shapley's "efficiency"
    axiom). If grouping one-hot dummy columns back into original
    feature names ever breaks that, this feature becomes misleading
    rather than useful - so this is a hard requirement, not a nice-to-have.
    """
    shap_row = [0.5, -0.2, 0.0, 1.3, -0.7, 2.1]
    feature_names = [
        "categorical__Make_BMW", "categorical__Make_Honda", "categorical__Make_Toyota",
        "categorical__Cylinders_4", "numeric__mileage", "numeric__Age",
    ]
    grouped = _group_shap_by_original_feature(shap_row, feature_names)

    assert np.isclose(sum(grouped.values()), sum(shap_row))
    assert "Make" in grouped and "Cylinders" in grouped
    assert np.isclose(grouped["Make"], 0.5 - 0.2 + 0.0)