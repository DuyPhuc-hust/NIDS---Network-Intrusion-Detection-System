import joblib
import numpy as np
import pandas as pd

from src.utils.config import MODEL_DIR

model = joblib.load(f"{MODEL_DIR}/model.pkl")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
features = joblib.load(f"{MODEL_DIR}/features.pkl")

def align_features(df):
    for col in features:
        if col not in df.columns:
            df[col] = 0
    return df[features]

def predict(input_df):
    input_df = align_features(input_df)
    X = scaler.transform(input_df)
    return model.predict(X)