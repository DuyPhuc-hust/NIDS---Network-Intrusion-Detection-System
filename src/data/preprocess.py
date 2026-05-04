import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def clean_data(df):
    print("[+] Cleaning data...")

    # fix tên cột (CICIDS hay bị space)
    df.columns = df.columns.str.strip()

    # xử lý label nếu có
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip()

    # replace inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # drop NaN
    df.dropna(inplace=True)

    print(f"[+] After cleaning: {df.shape}")
    return df


def split_xy(df, label_col="Label"):
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


def encode_features_train_test(X_train, X_test):
    categorical_cols = X_train.select_dtypes(include=["object"]).columns

    if len(categorical_cols) == 0:
        return X_train, X_test, None

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_train_cat = encoder.fit_transform(X_train[categorical_cols])
    X_test_cat = encoder.transform(X_test[categorical_cols])

    cat_columns = encoder.get_feature_names_out(categorical_cols)

    X_train_cat = pd.DataFrame(X_train_cat, columns=cat_columns, index=X_train.index)
    X_test_cat = pd.DataFrame(X_test_cat, columns=cat_columns, index=X_test.index)

    # drop old
    X_train = X_train.drop(columns=categorical_cols)
    X_test = X_test.drop(columns=categorical_cols)

    # concat lại → vẫn là DataFrame
    X_train = pd.concat([X_train, X_train_cat], axis=1)
    X_test = pd.concat([X_test, X_test_cat], axis=1)

    return X_train, X_test, encoder


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    # đảm bảo là DataFrame
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, scaler