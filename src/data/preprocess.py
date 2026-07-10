import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def clean_data(df):
    """
    Clean CICIDS-style flow data for training or inference.
    The function automatically handles labeled training data and unlabeled PCAP-derived flows.
    """
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()

    drop_cols = [
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp',
        'flow_id', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'timestamp'
    ]
    
    existing_drop = [c for c in drop_cols if c in df_clean.columns]
    df_clean.drop(columns=existing_drop, inplace=True)

    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.fillna(0, inplace=True)

    if "Label" in df_clean.columns or "label" in df_clean.columns:
        label_col = "Label" if "Label" in df_clean.columns else "label"
        
        label_mapping = {
            "BENIGN": "Normal",
            "Benign": "Normal",
            "Normal": "Normal",
            "DoS Hulk": "DoS",
            "DoS GoldenEye": "DoS",
            "DoS Slowloris": "DoS",
            "DoS slowloris": "DoS",
            "DoS Slowhttptest": "DoS",
            "DoS": "DoS",
            "DDoS": "DDoS",
            "PortScan": "PortScan",
            "FTP-Patator": "BruteForce",
            "SSH-Patator": "BruteForce",
            "BruteForce": "BruteForce",
            "Bot": "Bot",
            "Web Attack  Brute Force": "Web",
            "Web Attack  XSS": "Web",
            "Web Attack  Sql Injection": "Web",
            "Web": "Web",
            "Infiltration": "Infiltration",
            "Heartbleed": "Infiltration"
        }

        df_clean[label_col] = df_clean[label_col].astype(str).str.strip().map(label_mapping)
        df_clean.dropna(subset=[label_col], inplace=True)

        print("\n[+] Class distribution after label grouping:")
        print(df_clean[label_col].value_counts())
    else:
        label_col = None
        print("[*] Live/inference data detected: no Label column found.")

    feature_cols = [c for c in df_clean.columns if c != label_col]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.fillna(0, inplace=True)

    return df_clean


def split_xy(df):
    """
    Split features and labels.
    """
    if "Label" in df.columns:
        X = df.drop("Label", axis=1)
        y = df["Label"]
    elif "label" in df.columns:
        X = df.drop("label", axis=1)
        y = df["label"]
    else:
        X = df
        y = None
    return X, y


def handle_imbalance(X_train, y_train):
    """
    Conservative imbalance handling.
    Normal traffic is under-sampled, while minority classes are only lightly over-sampled.
    Extremely small classes are not forced through aggressive SMOTE because synthetic samples
    from tiny classes can make evaluation look better than real-world generalization.
    """
    print("\nConservative imbalance handling")

    current_counts = y_train.value_counts()
    normal_count = current_counts.get('Normal', 0)
    target_normal = min(300000, normal_count)

    if normal_count > target_normal:
        rus = RandomUnderSampler(
            sampling_strategy={'Normal': target_normal},
            random_state=42
        )
        X_train, y_train = rus.fit_resample(X_train, y_train)
        current_counts = y_train.value_counts()

    smote_strategy = {}
    for label, count in current_counts.items():
        if label == 'Normal':
            continue

        # Conservative SMOTE: avoid manufacturing too many synthetic samples from
        # tiny classes because that can make the test results look better than the
        # model's real-world generalization.
        if count < 100:
            target = count
        elif label == 'Bot':
            target = min(int(count * 1.5), 2500)
        elif count < 2000:
            target = min(int(count * 1.5), 4000)
        elif count < 10000:
            target = min(int(count * 1.25), 12000)
        else:
            target = count
        smote_strategy[label] = max(count, target)

    smote_strategy = {
        label: target
        for label, target in smote_strategy.items()
        if current_counts.get(label, 0) >= 2 and target > current_counts.get(label, 0)
    }

    if smote_strategy:
        min_class_count = min(current_counts[label] for label in smote_strategy)
        smote = SMOTE(
            sampling_strategy=smote_strategy,
            random_state=42,
            k_neighbors=min(5, min_class_count - 1)
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)
    else:
        print("[!] Skipping SMOTE because no class has enough samples for synthetic oversampling.")
        X_res, y_res = X_train, y_train

    print(f"After imbalance handling: {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res


def encode_labels(y_train, y_test, y_val=None):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    if y_val is not None:
        y_val_enc = le.transform(y_val)
        return y_train_enc, y_test_enc, y_val_enc, le

    return y_train_enc, y_test_enc, le


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
