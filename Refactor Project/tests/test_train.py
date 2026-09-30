import pandas as pd
from AutoDealer.train import train_and_evaluate
import numpy as np
def test_train():
    result=train_and_evaluate()
    r2_gap = result["TRAIN_R2"] - result["R^2"]
    RMSE_gap = result["RMSE"] - result["Train_RMSE"]
    assert r2_gap < 0.05, f"Train/test R2 gap too large: {r2_gap:.3f}"
    assert RMSE_gap < 2000, f"Train/test RMSE gap too large:{RMSE_gap}"
