"""
test_preprocess.py

Unit tests for preprocess.py. Most functions here take and return
plain DataFrames, so they're tested with small hand-built inputs rather
than the real master.csv - faster, and each test isolates exactly one
transformation.

Known gap: clean_master_columns splits make_model naively on the first
word (Year) and second word (Make). Multi-word makes like "Land Rover"
or "Aston Martin" aren't handled correctly yet - that's a known,
unresolved design decision, not something these tests pretend to cover.
"""

from dataclasses import replace

import pandas as pd
import pytest

from AutoDealer import preprocess
from AutoDealer.config import CONFIG
from AutoDealer.preprocess import (
    apply_std_tier,
    build_std_tier_lookup,
    clean_master_columns,
    compute_age,
    compute_is_hybrid,
    drop_high_nan_columns,
    drop_rare_makes,
    drop_unused_columns,
    filter_by_year,
    filter_rows,
    load_master,
    merge_engine_specs,
)


# ---------------------------------------------------------------------------
# clean_master_columns
# ---------------------------------------------------------------------------

def test_clean_master_columns():
    raw = pd.DataFrame({
        "make_model": ["2026 Toyota Camry", "2015 Honda Civic",
                       "2027 BMW X5", "2024 Ford Escape"],
        "trim": ["LE", "EX", "xDrive", "SE"],
        "price": ["$47,998", "$12,500", "$65,000", "Price on Request"],
        "mileage": ["0 km", "120,000 km", "300 km", "40,000 km"],
        "transmission": ["Automatic", "Manual", "Automatic", "Automatic"],
        "fuel_type": ["Gas", "Gas", "Gas/Electric Hybrid", "Gas"],
    })

    result = clean_master_columns(raw)

    # the "Price on Request" row should be dropped entirely
    assert len(result) == 3

    assert list(result["Make"]) == ["Toyota", "Honda", "BMW"]
    assert list(result["Model"]) == ["Camry", "Civic", "X5"]
    assert list(result["Year"]) == [2026, 2015, 2027]
    assert list(result["price"]) == [47998.0, 12500.0, 65000.0]
    assert list(result["mileage"]) == [0.0, 120000.0, 300.0]

    # 2026/2027 are within CONFIG.current_year's New-car window and have
    # low mileage; 2015 is a Used car regardless of mileage
    assert list(result["Condition"]) == ["New", "Used", "New"]


# ---------------------------------------------------------------------------
# compute_is_hybrid
# ---------------------------------------------------------------------------

def test_compute_is_hybrid():
    df = pd.DataFrame({"fuel_type": ["Gas", "Gas/Electric Hybrid", None]})
    result = compute_is_hybrid(df)
    assert list(result["Is_Hybrid"]) == [0, 1, 0]


# ---------------------------------------------------------------------------
# build_std_tier_lookup - the leakage regression test
# ---------------------------------------------------------------------------

def test_std_tier_lookup_uses_only_new_rows():
    """
    This is the key regression test for the leakage fix: a Used row's
    price must never influence the std_tier lookup. We prove this by
    computing the lookup with and without an extreme-priced Used row
    added, and asserting the result is identical either way.
    """
    new_only = pd.DataFrame({
        "Make": ["Toyota", "Toyota", "Toyota", "Toyota"],
        "Model": ["Camry", "Camry", "Supra", "Supra"],
        "price": [30000, 31000, 60000, 61000],
        "Condition": ["New", "New", "New", "New"],
    })

    with_used_outlier = pd.concat([
        new_only,
        pd.DataFrame({
            "Make": ["Toyota"], "Model": ["Camry"],
            "price": [999999], "Condition": ["Used"],
        }),
    ], ignore_index=True)

    lookup_without_used = build_std_tier_lookup(new_only)
    lookup_with_used = build_std_tier_lookup(with_used_outlier)

    pd.testing.assert_frame_equal(
        lookup_without_used.sort_values(["Make", "Model"]).reset_index(drop=True),
        lookup_with_used.sort_values(["Make", "Model"]).reset_index(drop=True),
    )


def test_apply_std_tier_defaults_to_normal_for_unmatched():
    lookup = pd.DataFrame({"Make": ["Toyota"], "Model": ["Camry"], "std_tier": ["2std"]})
    df = pd.DataFrame({"Make": ["Toyota", "Honda"], "Model": ["Camry", "Civic"]})

    result = apply_std_tier(df, lookup)

    assert result.loc[result["Model"] == "Camry", "std_tier"].iloc[0] == "2std"
    assert result.loc[result["Model"] == "Civic", "std_tier"].iloc[0] == "Normal"


# ---------------------------------------------------------------------------
# compute_age
# ---------------------------------------------------------------------------

def test_compute_age():
    df = pd.DataFrame({"Year": [2020, 2015]})
    result = compute_age(df, current_year=2027)
    assert list(result["Age"]) == [7, 12]


# ---------------------------------------------------------------------------
# filter_rows
# ---------------------------------------------------------------------------

def test_filter_rows():
    df = pd.DataFrame({
        "Condition": ["Used", "Used", "New", "Used"],
        "Make": ["Toyota", "Porsche", "Toyota", "Honda"],
        "Cylinders": ["4", "6", "4", "Electric"],
        "trim": ["LE", "GTS", "LE", "EX"],
    })

    result = filter_rows(df)

    # Condition and trim should no longer exist
    assert "Condition" not in result.columns
    assert "trim" not in result.columns

    # Porsche dropped (excluded make), Honda dropped (Electric cylinder),
    # the New Toyota row dropped (not Used) - only the Used Toyota survives
    assert set(result["Make"]) == {"Toyota"}
    assert len(result) == 1


# ---------------------------------------------------------------------------
# filter_by_year
# ---------------------------------------------------------------------------

def test_filter_by_year():
    df = pd.DataFrame({"Year": [2010, 2000, 2023]})
    result = filter_by_year(df, min_year=2005)
    assert list(result["Year"]) == [2010, 2023]


# ---------------------------------------------------------------------------
# drop_rare_makes
# ---------------------------------------------------------------------------

def test_drop_rare_makes():
    df = pd.DataFrame({"Make": ["Toyota"] * 5 + ["Daihatsu"]})
    result = drop_rare_makes(df, min_count=3)
    assert "Daihatsu" not in result["Make"].values
    assert len(result) == 5


# ---------------------------------------------------------------------------
# drop_unused_columns
# ---------------------------------------------------------------------------

def test_drop_unused_columns():
    df = pd.DataFrame({
        "Model": [1], "fuel_type": [1], "Year": [1],
        "Country_of_Origin": [1], "Make": [1],
    })
    result = drop_unused_columns(df)
    assert set(result.columns) == {"Make"}


# ---------------------------------------------------------------------------
# drop_high_nan_columns
# ---------------------------------------------------------------------------

def test_drop_high_nan_columns():
    df = pd.DataFrame({
        "good": [1, 2, 3, 4],
        "bad": [1, None, None, None],
    })
    result = drop_high_nan_columns(df, threshold=10.0)
    assert "bad" not in result.columns
    assert "good" in result.columns


# ---------------------------------------------------------------------------
# drop_incomplete_rows - uses a monkeypatched CONFIG so the test doesn't
# depend on the real (large) feature list in config.py
# ---------------------------------------------------------------------------

def test_drop_incomplete_rows(monkeypatch):
    small_config = replace(
        CONFIG,
        target="price",
        numeric_features=("mileage",),
        categorical_features=("Make",),
    )
    monkeypatch.setattr(preprocess, "CONFIG", small_config)

    df = pd.DataFrame({
        "price": [20000, None, 25000],
        "mileage": [50000, 30000, None],
        "Make": ["Toyota", "Honda", "Ford"],
    })

    result = preprocess.drop_incomplete_rows(df)

    assert len(result) == 1
    assert result.iloc[0]["Make"] == "Toyota"


# ---------------------------------------------------------------------------
# load_master - file I/O, uses tmp_path so the real data/ folder is
# never touched
# ---------------------------------------------------------------------------

def test_load_master_raises_when_file_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_master(missing_path)


def test_load_master_loads_existing_file(tmp_path):
    path = tmp_path / "master.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)

    result = load_master(path)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# merge_engine_specs - file I/O, uses a temp xlsx so the real
# EngineType_enriched.xlsx is never touched
# ---------------------------------------------------------------------------

def test_merge_engine_specs(tmp_path):
    spec_path = tmp_path / "engine.xlsx"
    pd.DataFrame({
        "Make": ["Toyota", "Toyota", "Toyota"],
        "Model": ["Camry", "Camry", "Camry"],
        "Cylinders": ["4", "4", "6"],
        "Body Type": ["Sedan", "Sedan", "Sedan"],
    }).to_excel(spec_path, index=False)

    df = pd.DataFrame({"Make": ["Toyota", "Honda"], "Model": ["Camry", "Civic"]})
    result = merge_engine_specs(df, engine_spec_path=spec_path)

    camry = result[result["Model"] == "Camry"].iloc[0]
    assert camry["Cylinders"] == "4"  # mode of [4, 4, 6] is 4

    civic = result[result["Model"] == "Civic"].iloc[0]
    assert civic["Cylinders"] == "Unknown"  # no match in the spec file