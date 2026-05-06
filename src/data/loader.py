import os
import pandas as pd

def load_cicids(data_path, sample_size=None):
    dfs = []

    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            path = os.path.join(data_path, file)
            print(f"[+] Loading {file}")
            df = pd.read_csv(path)

            # FIX tên cột (có khoảng trắng)
            df.columns = df.columns.str.strip()

            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    print(f"[+] Total shape: {df.shape}")
    return df