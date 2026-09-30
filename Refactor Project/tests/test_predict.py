import joblib
import pytest

from AutoDealer.pipeline import build_pipeline
from AutoDealer.predict import predict


def test_predict_returns_one_value_per_row(tmp_path, sample_data):
    X, y = sample_data
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    model_file = tmp_path / "model.joblib"   # temp folder, real model untouched
    joblib.dump(pipeline, model_file)

    predictions = predict(X, model_path=model_file)  # X has no price column
    assert len(predictions) == len(X)


def test_predict_raises_when_model_missing(tmp_path, sample_data):
    X, _ = sample_data
    with pytest.raises(FileNotFoundError):
        predict(X, model_path=tmp_path / "does_not_exist.joblib")