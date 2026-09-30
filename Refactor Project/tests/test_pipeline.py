import pandas as pd
from AutoDealer.pipeline import build_pipeline

def test_pipeline_fits_and_predicts():
    X_train = pd.DataFrame({
        "Make": ["Toyota", "Honda", "Toyota", "Ford"],
        "transmission": ["Automatic", "Manual", "Automatic", "Automatic"],
        "Country_of_Origin": ["Japan", "Japan", "Japan", "USA"],
        "Body Type": ["Sedan", "SUV", "Sedan", "Truck"],
        "Cylinders": ["4", "4", "6", "8"],
        "Brand_Segment": ["Mainstream", "Mainstream", "Mainstream", "Mainstream"],
        "std_tier": ["Normal", "Normal", "1std", "Normal"],
        "mileage": [50000, 30000, 20000, 80000],
        "Age": [3, 2, 1, 5],
        "no_accidents": [1, 0, 1, 1],
        "has_carfax": [1, 1, 0, 1],
        "one_owner": [1, 0, 1, 0],
        "service_records": [0, 1, 0, 0],
        "certified": [0, 0, 1, 0],
        "Is_Hybrid": [0, 0, 0, 0],
    })
    y_train = pd.Series([20000, 18000, 25000, 30000])

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_train)
    assert len(predictions) == len(X_train)


def test_pipeline_handles_unseen_category_gracefully():
    """This is the key regression test for the std_tier bug class."""
    X_train = pd.DataFrame({
        "Make": ["Toyota", "Honda"],
        "transmission": ["Automatic", "Manual"],
        "Country_of_Origin": ["Japan", "Japan"],
        "Body Type": ["Sedan", "SUV"],
        "Cylinders": ["4", "4"],
        "Brand_Segment": ["Mainstream", "Mainstream"],
        "std_tier": ["Normal", "Normal"],
        "mileage": [50000, 30000],
        "Age": [3, 2],
        "no_accidents": [1, 0],
        "has_carfax": [1, 1],
        "one_owner": [1, 0],
        "service_records": [0, 1],
        "certified": [0, 0],
        "Is_Hybrid": [0, 0],
    })
    y_train = pd.Series([20000, 18000])

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    X_new = X_train.copy()
    X_new.loc[0, "Make"] = "Ferrari"  # never seen during training

    predictions = pipeline.predict(X_new)  # should NOT raise
    assert len(predictions) == 2