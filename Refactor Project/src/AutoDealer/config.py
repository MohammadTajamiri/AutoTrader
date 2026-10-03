from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  


@dataclass(frozen=True)
class Config:
    # paths
    std_tier_source_path: Path = ROOT / "data" / "FullDataSet.csv"
    std_tier_reference_path: Path = ROOT / "data" / "StdTierReference.csv"
    raw_data_path: Path = ROOT / "data" / "ModelInput.csv"
    new_input_path: Path = ROOT / "data" / "NewInput.csv"
    raw_master_path: Path = ROOT / "data" / "master.csv"
    engine_spec_path: Path = ROOT / "data" / "EngineType_enriched.xlsx"
    model_path: Path = ROOT / "models" / "model.joblib"
    std_tier_reference_path: Path = ROOT / "data" / "StdTierReference.csv"
    error_model_path: Path = ROOT / "models" / "error_model.joblib"
    # columns
    target: str = "price"
    numeric_features: tuple[str, ...] = (
        "mileage", "Age",
        "no_accidents", "has_carfax", "one_owner",
        "service_records", "certified", "Is_Hybrid",
    )
    categorical_features: tuple[str, ...] = (
    "Make", "transmission",
    "Body Type", "Cylinders", "Brand_Segment", "std_tier",
)

    # preprocessing
    min_manufacture_year: int = 2005
    current_year: int = 2027
    price_bin_width: int = 5000
    min_price_bin_count: int = 100
    excluded_makes: tuple[str, ...] = ("Land", "Porsche")
    excluded_cylinders: tuple[str, ...] = ("Electric",)

    # split and cv
    test_size: float = 0.3
    n_splits: int = 5
    random_state: int = 90

    # winning hyperparameters
    model_params: dict = field(default_factory=lambda: {
        "n_estimators": 3000,
        "max_depth": 4,
        "learning_rate": 0.03,
    })
CONFIG = Config()