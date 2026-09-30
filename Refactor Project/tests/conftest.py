import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    X = pd.DataFrame({
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
    y = pd.Series([20000, 18000, 25000, 30000])
    return X, y