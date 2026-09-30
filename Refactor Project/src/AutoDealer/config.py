from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  


@dataclass(frozen=True)
class Config:
    # paths
    raw_data_path: Path = ROOT / "data" / "TreeInput.csv"
    model_path: Path = ROOT / "models" / "model.joblib"

    # columns
    target: str = "price"
    numeric_features: tuple[str, ...] = (
        "mileage", "Age",
        "no_accidents", "has_carfax", "one_owner",
        "service_records", "certified", "Is_Hybrid",
    )
    categorical_features: tuple[str, ...] = (
        "Make", "transmission", "Country_of_Origin",
        "Body Type", "Cylinders", "Brand_Segment", "std_tier",
    )

    # split and cv
    test_size: float = 0.2
    n_splits: int = 5
    random_state: int = 42

    # winning hyperparameters from your tuning run
    model_params: dict = field(default_factory=lambda: {
        "n_estimators": 4000,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample":0.8,
        "n_jobs":-1
    })


CONFIG = Config()