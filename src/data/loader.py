import pandas as pd
import os

def load_cicids(data_path, sample_size=None):
    all_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

    df_list = []

    for file in all_files:
        file_path = os.path.join(data_path, file)
        print(f"[+] Loading {file}")

        df = pd.read_csv(file_path)

        # ✅ FIX 1: strip toàn bộ column
        df.columns = df.columns.str.strip()

        # ✅ FIX 2: rename nếu có biến thể
        if " Label" in df.columns:
            df.rename(columns={" Label": "Label"}, inplace=True)

        if "\ufeffLabel" in df.columns:
            df.rename(columns={"\ufeffLabel": "Label"}, inplace=True)

        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    print(f"[+] Total shape: {df.shape}")

    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    return df