import joblib
import pandas as pd
from src.pipeline import full_preprocess

def load_model(model_path="models/xgb_churn_model.pkl"):
    return joblib.load(model_path)

def predict_churn(input_df, model):
    """
    Predict churn from a raw input DataFrame.

    Steps:
    - preprocess input using full_preprocess
    - return predicted class and probability
    """
    processed = full_preprocess(input_df.copy())
    pred = model.predict(processed)
    prob = model.predict_proba(processed)[:, 1]  # churn probability

    return pred, prob