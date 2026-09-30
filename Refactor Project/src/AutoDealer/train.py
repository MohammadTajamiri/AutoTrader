"""
train.py

Loads data, fits the pipeline, evaluates it, and saves the trained
pipeline artifact to disk.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from .config import CONFIG
from .data import load_and_split
from .pipeline import build_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate():
    #  1: get X_train, X_test, y_train, y_test from data.py's
    # load_and_split()
    X_train,X_test,y_train,y_test=load_and_split()
    #  2: build an unfitted pipeline from pipeline.py
    pipeline = build_pipeline()
    #  3: fit the pipeline on the training data
    pipeline.fit(X_train,y_train)
    #  4: generate predictions on X_test (and optionally X_train,
    # if you want train vs. test comparison like your notebook did)
    y_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)
    # 5: compute at least RMSE and R² on the test predictions
    #   - hint: mean_squared_error gives you MSE; how do you get RMSE
    #     from that with numpy?
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    train_r2   = r2_score(y_train, y_train_pred)
    train_mse  = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    # 6: print or return these metrics so you can see how the
    # model did
    result = {
            "MAE":np.round(mae,3),"MSE":np.round(mse,3),"RMSE":np.round(rmse,3),"Train_RMSE":np.round(train_rmse,2),"R^2":np.round(r2,3),"TRAIN_R2":np.round(train_r2,3)
        }
    # 7: save the FITTED pipeline (not the unfitted one from step 2)
    # to CONFIG.model_path using joblib.dump()
    #   - hint: does CONFIG.model_path's parent directory definitely
    #     exist? what happens if it doesn't?
    CONFIG.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, CONFIG.model_path)

    return result


if __name__ == "__main__":
    result=train_and_evaluate()
    print(f"MAE is {result["MAE"]} , RMSE is {result["RMSE"]} ,TRAIN RMSE is {result["Train_RMSE"]}, R_2 is {result["R^2"]} , Train_R_2 is {result["TRAIN_R2"]}")