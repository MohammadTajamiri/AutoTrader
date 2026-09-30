"""
pipeline.py

Builds the full preprocessing + model pipeline as a single sklearn
Pipeline object.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .config import CONFIG


def build_pipeline() -> Pipeline:
    """
    Returns an unfitted sklearn Pipeline: encoding step + XGBRegressor.
    """
    # TODO 1: Build a ColumnTransformer with two transformers:
    #   - one named "categorical" that applies OneHotEncoder to
    #     CONFIG.categorical_features
    #       -> pass handle_unknown="ignore" to the encoder. Think about
    #          why: what should happen if the model sees a category
    #          at prediction time that it never saw during training?
    #   - one named "numeric" that just passes CONFIG.numeric_features
    #     through unchanged (hint: ColumnTransformer accepts the string
    #     "passthrough" as a transformer)
    #  
    # preprocessor = ColumnTransformer(transformers=[...])
    preprocess = ColumnTransformer(
        transformers=[("number",'passthrough',CONFIG.numeric_features),("categoris",OneHotEncoder(handle_unknown='ignore'),CONFIG.categorical_features)]
    )
    # TODO 2: Build an XGBRegressor using CONFIG.model_params.
    #   - CONFIG.model_params is a dict — how do you unpack a dict
    #     into keyword arguments when calling a function/constructor?
    #   - also pass CONFIG.random_state explicitly, since model_params
    #     doesn't currently include it and you want reproducibility
    #     matched to your train/test split
    #
    model = XGBRegressor(**CONFIG.model_params,random_state =CONFIG.random_state )

    # TODO 3: Chain preprocessor -> model into one Pipeline.
    #   - sklearn's Pipeline takes a `steps` list of (name, transformer)
    #     tuples. What two steps go in here, in what order?
    #
    pipeline = Pipeline(steps=[('preprocess',preprocess),('model',model)])

    # TODO 4: return the pipeline
    return pipeline