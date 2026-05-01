import pandas as pd
import os

def load_cicids(folder_path, sample_size=None):
    dfs = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            path = os.path.join(folder_path, file)
            print(f"[+] Loading {file}")

            if sample_size:
                df = pd.read_csv(path, nrows=sample_size)
            else:
                df = pd.read_csv(path, low_memory=False)

            dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[+] Total shape: {df_all.shape}")

    return df_all