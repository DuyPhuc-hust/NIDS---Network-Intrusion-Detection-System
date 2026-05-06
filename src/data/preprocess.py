import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def clean_data(df):
    """
    Làm sạch dữ liệu, xử lý giá trị vô hạn.
    Tự động phân biệt giữa chế độ Train (có nhãn) và Live (không nhãn).
    """
    # 1. Tạo bản sao và chuẩn hóa tên cột
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()

    # 2. Loại bỏ các cột Metadata (Thông tin định danh không dùng để train)
    drop_cols = [
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp',
        'flow_id', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'timestamp'
    ]
    
    # Chỉ xóa nếu cột đó tồn tại
    existing_drop = [c for c in drop_cols if c in df_clean.columns]
    df_clean.drop(columns=existing_drop, inplace=True)

    # 3. Xử lý giá trị vô hạn (inf) và giá trị thiếu (NaN)
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Thay vì dropna ngay, ta fillna(0) để tránh mất dữ liệu khi live analyze
    df_clean.fillna(0, inplace=True)

    # 4. 🔥 XỬ LÝ NHÃN (Chỉ thực hiện nếu có cột Label - Phục vụ Training)
    if "Label" in df_clean.columns or "label" in df_clean.columns:
        label_col = "Label" if "Label" in df_clean.columns else "label"
        
        # Gom nhóm các cuộc tấn công
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
            "Web Attack  Brute Force": "Web",
            "Web Attack  XSS": "Web",
            "Web Attack  Sql Injection": "Web",
            "Infiltration": "Infiltration",
            "Heartbleed": "Misc"
        }

        # Áp dụng map nhãn
        df_clean[label_col] = df_clean[label_col].map(label_mapping)

        # Nếu là lúc Train, ta xóa các dòng không map được (unknown)
        # Nhưng nếu là lúc Live, ta không làm bước này
        df_clean.dropna(subset=[label_col], inplace=True)

        print("\n[+] Thống kê các lớp sau khi gom nhóm:")
        print(df_clean[label_col].value_counts())
    else:
        # Nếu không có cột Label, in thông báo (Chế độ Live Analyze)
        print("[*]")

    return df_clean

def split_xy(df):
    """
    Tách đặc trưng (X) và nhãn (y).
    """
    if "Label" in df.columns:
        X = df.drop("Label", axis=1)
        y = df["Label"]
    elif "label" in df.columns:
        X = df.drop("label", axis=1)
        y = df["label"]
    else:
        # Trường hợp live analyze không có nhãn
        X = df
        y = None
    return X, y

def handle_imbalance(X_train, y_train):
    """
    Chiến lược Hybrid: Under-sampling lớp Normal và SMOTE lớp thiểu số.
    """
    print("\nSMOTE + Under-sampling")
    
    # Under-sampling lớp Normal xuống mức 500,000 mẫu
    under_strategy = {'Normal': 500000}
    current_normal = (y_train == 'Normal').sum()
    
    if current_normal > 500000:
        rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)

    # SMOTE các lớp còn lại
    smote = SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"Sau xử lý: {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res

def encode_labels(y_train, y_test):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc, le

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler