"""
preprocess.py

Turns a raw scraped master.csv into the same shape as TreeInput.csv,
so train.py can be pointed at either one.

Order matters here. In particular: std_tier must be computed from
Condition == "New" rows BEFORE those rows are dropped, and that
computation must never use a Used row's own price - that's the
leakage bug this pipeline specifically avoids.
"""

import pandas as pd
import numpy as np
from .config import CONFIG
from .enrichment import enrich_dataframe


def load_master(path=CONFIG.raw_master_path) -> pd.DataFrame:
    """Loads the raw scraped file. Falls back to a more permissive
    encoding if the file isn't plain UTF-8 - common when a CSV has been
    exported from Excel on Windows."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}. "
            f"Check that the file exists and CONFIG.raw_master_path is correct."
        )
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def clean_master_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns the raw scraped master.csv columns into the shape the rest of
    the pipeline expects: Make, Model, Year as separate columns, price
    and mileage as real floats, and a Condition flag.

    Does NOT compute Age - that's compute_age()'s job, kept separate so
    the "current year" constant lives in exactly one place (CONFIG).
    """
    new_car_years = [CONFIG.current_year, CONFIG.current_year - 1, CONFIG.current_year - 2]
    df = df.drop(columns=["scraped_at", "source_url", "page", "url",
                           "Dealer Name", "Dealer Address", "Dealer URL"],
                 errors="ignore")

    # "2022 Lincoln Corsair" -> Year="2022", Make="Lincoln", Model="Corsair"
    split = df["make_model"].str.split(" ", expand=True, n=2)
    df["Year"] = pd.to_numeric(split[0], errors="coerce")
    df["Make"] = split[1]
    df["Model"] = split[2]
    df = df.drop(columns=["make_model"])

    df = df[df["price"] != "Price on Request"].copy()
    df["price"] = (
        df["price"].str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .astype(float)
    )

    df["mileage"] = (
        df["mileage"].str.replace("km", "", regex=False)
                      .str.replace(",", "", regex=False)
                      .astype(float)
    )

    # Condition needs Year and mileage, computed against CONFIG's
    # current year - not Age, which doesn't exist yet at this point.
    df["Condition"] = np.where(
        (df["Year"].isin(new_car_years)) & (df["mileage"] <= 450),
        "New", "Used"
    )
    return df
def apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops excluded Makes and Cylinders (CONFIG.excluded_makes,
    CONFIG.excluded_cylinders) - shared by the Used-car training
    pipeline and the std_tier reference build, so both datasets apply
    the same brand/engine exclusions consistently.
    """
    df = df[~df["Make"].isin(CONFIG.excluded_makes)]
    df = df[~df["Cylinders"].isin(CONFIG.excluded_cylinders)]
    return df
def drop_rare_makes(df: pd.DataFrame, min_count: int = 30) -> pd.DataFrame:
    """
    Drops any Make with fewer than min_count rows. Needed because
    train_test_split's stratify=Make requires every category to have
    at least 2 rows (one for train, one for test) - a Make with only
    1 listing in the whole dataset breaks the split entirely. A higher
    min_count also means the model actually has enough examples to
    learn something about that brand, rather than just satisfying the
    stratify requirement at the bare minimum.
    """
    make_counts = df["Make"].value_counts()
    rare_makes = make_counts[make_counts < min_count].index.tolist()

    if rare_makes:
        print(f"Dropping {len(rare_makes)} makes with fewer than {min_count} rows: {rare_makes}")
        df = df[~df["Make"].isin(rare_makes)]

    return df
def fill_missing_transmission(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing transmission values with "Automatic", the most
    common value in the dataset. Better than dropping these rows
    outright, since a missing transmission value doesn't mean the row
    is otherwise bad data.
    """
    df = df.copy()
    missing_count = df["transmission"].isna().sum()
    if missing_count:
        print(f"Filling {missing_count} missing transmission values with 'Automatic'.")
        df["transmission"] = df["transmission"].fillna("Automatic")
    return df
def drop_sparse_price_bins(df: pd.DataFrame, bin_width: int = 5000, min_bin_count: int = 100) -> pd.DataFrame:
    """
    Drops rows whose price falls in a sparsely-populated price bracket.
    This is a deliberate scope-narrowing decision, not a bug fix: it
    disproportionately removes expensive/luxury cars, since they're
    naturally rarer in any scrape than mainstream vehicles. The
    resulting model's reported RMSE/R^2 reflect performance on the
    price ranges that remain, NOT on the full market - see
    docs/decisions.md for the reasoning.
    """
    price_bins = pd.cut(df["price"], bins=range(0, int(df["price"].max()) + bin_width, bin_width))
    bin_counts = price_bins.value_counts()
    valid_bins = bin_counts[bin_counts >= min_bin_count].index

    before = len(df)
    df = df[price_bins.isin(valid_bins)].copy()
    print(f"Dropped {before - len(df)} rows in sparse price brackets (<{min_bin_count} cars per ${bin_width} bin). Kept {len(df)}.")
    return df
def _model_merge_key(model_series: pd.Series) -> pd.Series:
    """
    EngineType_enriched.xlsx names a car down to its FIRST WORD only
    ("RAV", "Grand", "3", "Civic") - whatever came after that ("4",
    "Series Sedan", "Cherokee") landed in that file's own
    MoreInformation column, not in its Model column. Our own Model
    column (from clean_master_columns) keeps the full name instead
    ("RAV 4", "Civic Sedan", "3 Series"). Joining on raw Model therefore
    misses every multi-word model - about 30% of rows. This reduces
    both sides to the same granularity: first word, case-folded, so
    "Civic Sedan" and "sierra" both line up with "Civic"/"Sierra".
    """
    return model_series.astype(str).str.strip().str.split(" ").str[0].str.upper()


def merge_engine_specs(df: pd.DataFrame, engine_spec_path=CONFIG.engine_spec_path) -> pd.DataFrame:
    """
    Merges Cylinders and Body Type onto df, using the mode (most common
    value) per (Make, Model) from the EngineType_enriched.xlsx reference
    file.

    IMPORTANT: the join key is a normalized first-word key, not the raw
    Model column - see _model_merge_key's docstring for why. This
    recovers ~7,870 of the ~7,894 rows that previously fell back to
    "Unknown" (30.3% -> ~0.1% of the dataset).

    Known limit, not a bug: a few Makes have two real models that share
    a first word (Jeep "Grand Cherokee" vs "Grand Wagoneer" both key to
    "Grand"). The reference file already conflates these into one row
    today, so this merge doesn't introduce new ambiguity - it just
    inherits what's already there.
    """
    engine_df = pd.read_excel(engine_spec_path)
    engine_df["Make"] = engine_df["Make"].astype(str)
    engine_df["Model"] = engine_df["Model"].astype(str)
    engine_df["Cylinders"] = engine_df["Cylinders"].astype(str).str.replace(r"\.0$", "", regex=True)
    engine_df["Model_key"] = _model_merge_key(engine_df["Model"])

    def safe_mode(series):
        m = series.mode()
        return m.iloc[0] if not m.empty else None

    # Group directly on the normalized key - NOT grouped by raw Model
    # and re-keyed afterward - otherwise two raw Models that normalize
    # to the same key would produce duplicate (Make, Model_key) rows
    # and silently blow up the merge below (the same duplicate-key bug
    # class that hit the enrichment merge earlier in this project).
    engine_lookup = engine_df.groupby(["Make", "Model_key"]).agg(
        Cylinders=("Cylinders", safe_mode),
        **{"Body Type": ("Body Type", safe_mode)}
    ).reset_index()

    df = df.copy()
    df["Make"] = df["Make"].astype(str)
    df["Model"] = df["Model"].astype(str)
    df["Model_key"] = _model_merge_key(df["Model"])

    df = df.merge(engine_lookup, on=["Make", "Model_key"], how="left", validate="many_to_one")
    df = df.drop(columns=["Model_key"])
    df["Cylinders"] = df["Cylinders"].fillna("Unknown")
    df["Body Type"] = df["Body Type"].fillna("Unknown")

    return df

def build_std_tier_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes (Make, Model) -> std_tier using ONLY Condition == "New"
    rows. This is leak-free because New and Used are structurally
    separate populations - a Used row's std_tier never depends on its
    own price or on other Used rows' prices.

    Models with no New-condition listings default to "Normal". A
    Used-price-based fallback was tried and reverted: any fallback
    computed from the Used population necessarily includes each row's
    own price when merged back onto that same row, reintroducing the
    original leakage in a less obvious form.
    """
    new_cars = df[df["Condition"] == "New"]

    rows = []
    for make, group in new_cars.groupby("Make"):
        make_mean = group["price"].mean()
        make_std = group["price"].std()
        model_avgs = group.groupby("Model")["price"].mean()

        for model, avg_price in model_avgs.items():
            if avg_price > make_mean + 3 * make_std:
                tier = "3std"
            elif avg_price > make_mean + 2 * make_std:
                tier = "2std"
            elif avg_price > make_mean + 1 * make_std:
                tier = "1std"
            else:
                tier = "Normal"
            rows.append({"Make": make, "Model": model, "std_tier": tier})

    return pd.DataFrame(rows, columns=["Make", "Model", "std_tier"])


def apply_std_tier(df: pd.DataFrame, reference_path=CONFIG.std_tier_reference_path) -> pd.DataFrame:
    """
    Left-merges std_tier onto every row of df via (Make, Model), reading
    the lookup from StdTierReference.csv - NOT recomputing it from
    whatever New-condition rows happen to be in df.

    This used to call build_std_tier_lookup(df) and compute std_tier
    from the same file being processed (master.csv for training,
    NewInput.csv for evaluation). That's the bug: a thin or absent
    New-car sample in any one file means a different population, a
    different mean/std, and a different std_tier for the SAME car
    depending only on which run processed it - confirmed concretely:
    BMW X7 came out as 2std during training vs 1std in
    StdTierReference.csv, and Audi SQ7 defaulted to "Normal" in both
    training and evaluation despite StdTierReference.csv (built from a
    dedicated, fuller New-car dataset) correctly flagging it 2std.

    StdTierReference.csv is built ONCE, separately, by
    build_std_tier_reference.py. Every run of preprocess.py now merges
    from that single file, so training and evaluation always agree,
    and rare/expensive models get a reliable tier even when a given
    run's own data has few or no New-condition examples of them.

    Rows with no match (a Model with no New-condition listings
    anywhere in the reference) default to "Normal".
    """
    if not reference_path.exists():
        raise FileNotFoundError(
            f"std_tier reference not found: {reference_path}. "
            f"Run `python -m AutoDealer.build_std_tier_reference` first."
        )
    lookup = pd.read_csv(reference_path)[["Make", "Model", "std_tier"]]
    df = df.merge(lookup, on=["Make", "Model"], how="left", validate="many_to_one")
    df["std_tier"] = df["std_tier"].fillna("Normal")
    return df

def compute_age(df: pd.DataFrame, current_year: int) -> pd.DataFrame:
    """Adds an Age column: current_year - Year."""
    df = df.copy()
    df["Age"] = current_year - df["Year"]
    return df


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only Used-condition rows, then drops Condition entirely since
    it has no further purpose once used cars are selected. Also applies
    the Make/Cylinders exclusions from CONFIG.
    """
    df = df[df["Condition"] == "Used"].copy()
    df = df.drop(columns=["Condition"])
    df = apply_exclusions(df)
    df = df.drop(columns=["trim"])
    return df

def compute_is_hybrid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the raw fuel_type column into a single binary Is_Hybrid
    flag. Electric-only cars were already dropped in filter_rows, so
    by the time this matters for training, the only distinction left
    worth keeping is Hybrid vs. everything else (Gas).
    """
    df = df.copy()
    df["Is_Hybrid"] = df["fuel_type"].str.contains("Hybrid", case=False, na=False).astype(int)
    return df

def report_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Returns the percentage of NaN values per column, sorted highest
    first. Call this before drop_high_nan_columns() to see exactly
    what's being dropped and why.
    """
    pct_missing = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    print(pct_missing[pct_missing > 0])
    return pct_missing
def drop_high_nan_columns(df: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """
    Drops any column whose missing-value rate exceeds threshold
    percent. Package_Description is the known case here - most listings
    don't mention a named package, so it's mostly NaN by design, not by
    data-quality error. Dropped for now; worth revisiting once there's
    a smarter way to treat "no package mentioned" as a real category
    rather than missing data.
    """
    pct_missing = df.isna().sum() / len(df) * 100
    cols_to_drop = pct_missing[pct_missing > threshold].index.tolist()

    if cols_to_drop:
        print(f"Dropping columns with >{threshold}% NaN: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    return df
def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows missing a value in any column the model actually uses."""
    required = [CONFIG.target, *CONFIG.numeric_features, *CONFIG.categorical_features]
    before = len(df)
    df = df.dropna(subset=required)
    if before != len(df):
        print(f"Dropped {before - len(df)} rows with missing required values (kept {len(df)} of {before}).")
    return df
def filter_by_year(df: pd.DataFrame, min_year: int) -> pd.DataFrame:
    """
    Drops cars manufactured before min_year. Older cars behave
    differently in the used market (collectibles, or just worn-out
    "shitboxes" as the project's own working notes call them) and
    don't fit the same price-vs-age-vs-mileage relationship the rest
    of this dataset is modeling.
    """
    before = len(df)
    df = df[df["Year"] >= min_year]
    print(f"Dropped {before - len(df)} cars made before {min_year} (kept {len(df)}).")
    return df

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that are no longer needed once their job is done:
    - Model: used for merges (engine specs, std_tier), not a training
      feature itself (same decision as the original notebook, which
      held Model out via lookup_cols rather than feeding it to the model)
    - fuel_type: fully replaced by Is_Hybrid, keeping the raw text adds
      nothing and was never a listed feature
    - transmission: dropped as a feature per project decision - also
      removed from config.py's categorical_features, since leaving it
      there would make validate_columns fail looking for a column that
      no longer exists
    """
    return df.drop(columns=["fuel_type","Year", "Country_of_Origin"], errors="ignore")
def build_tree_dataset(master_path=CONFIG.raw_master_path) -> pd.DataFrame:\
    
    df = load_master(master_path)
    df = clean_master_columns(df)
    df = filter_by_year(df, min_year=CONFIG.min_manufacture_year)
    df = drop_sparse_price_bins(df, bin_width=CONFIG.price_bin_width, min_bin_count=CONFIG.min_price_bin_count) 
    df = enrich_dataframe(df)
    df = merge_engine_specs(df)
    df = compute_is_hybrid(df)
    df = apply_std_tier(df)
    df = fill_missing_transmission(df)   
    report_missing_values(df)
    df = drop_high_nan_columns(df)
    df = compute_age(df, current_year=CONFIG.current_year)
    df = drop_rare_makes(df)
    df = filter_rows(df)
    df = drop_unused_columns(df)
    df = drop_incomplete_rows(df)
    return df

if __name__ == "__main__":
    df = build_tree_dataset()
    df.to_csv(CONFIG.raw_data_path, index=False)
    print(f"Wrote {len(df)} rows to {CONFIG.raw_data_path}")