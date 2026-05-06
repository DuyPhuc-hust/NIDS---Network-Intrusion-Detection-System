import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_data(df):
    import numpy as np

    df.columns = df.columns.str.strip()

    # Fix label column nếu có space
    if " Label" in df.columns:
        df.rename(columns={" Label": "Label"}, inplace=True)

    # Replace inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 🔥 MAP LABEL
    label_mapping = {
        "BENIGN": "Normal",
        "DoS Hulk": "DoS",
        "DoS GoldenEye": "DoS",
        "DoS Slowloris": "DoS",
        "DoS slowloris": "DoS",
        "DoS Slowhttptest": "DoS",
        "DDoS": "DDoS",
        "PortScan": "PortScan",
        "FTP-Patator": "BruteForce",
        "SSH-Patator": "BruteForce",
        "Bot": "Bot",
        "Web Attack � Brute Force": "Web",
        "Web Attack � XSS": "Web",
        "Web Attack � Sql Injection": "Web",
        "Infiltration": "Infiltration",
        "Heartbleed": "Misc"
    }

    df["Label"] = df["Label"].map(label_mapping)

    # Drop unknown (nếu có)
    df = df.dropna(subset=["Label"])

    print("\n[+] New classes:")
    print(df["Label"].value_counts())

    return df


def split_xy(df):
    X = df.drop("Label", axis=1)
    y = df["Label"]
    return X, y


def encode_labels(y_train, y_test):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    print("\n[+] Classes:")
    print(list(le.classes_))

    return y_train, y_test, le


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler