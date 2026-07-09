import os
import pandas as pd

def load_cicids(data_path, sample_size=None):
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    dfs = []
    csv_files = sorted(file for file in os.listdir(data_path) if file.endswith(".csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in dataset directory: {data_path}")

    for file in csv_files:
        path = os.path.join(data_path, file)
        print(f"[+] Loading {file}")
        df = pd.read_csv(path)

        # FIX tên cột (có khoảng trắng)
        df.columns = df.columns.str.strip()

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    if sample_size:
        if sample_size > len(df):
            print(f"[!] Requested sample_size={sample_size}, but dataset has {len(df)} rows. Using all rows.")
            sample_size = len(df)
        df = df.sample(n=sample_size, random_state=42)

    print(f"[+] Total shape: {df.shape}")
    return df
