import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.utils.config import *
from src.data.loader import load_csv
from src.data.preprocess import *

def train(data_path):
    print("[+] Loading data...")
    df = load_csv(data_path)

    print("[+] Cleaning...")
    df = clean_data(df, DROP_COLUMNS)

    print("[+] Splitting...")
    X, y = split_xy(df, TARGET_COLUMN)

    print("[+] Encoding...")
    X = encode_features(X)

    feature_columns = X.columns.tolist()

    print("[+] Scaling...")
    X_scaled, scaler = scale_features(X)

    print("[+] Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("[+] Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    print("[+] Evaluating...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("[+] Saving model...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, "features.pkl"))

    print("[+] Done!")