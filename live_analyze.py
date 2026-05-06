import os
import sys
import pandas as pd
import joblib
import subprocess
import time
import numpy as np
from src.data.preprocess import clean_data

# Configuration
MODEL_PATH = "models/xgb.pkl"
TEMP_CSV = "temp_extracted_flow.csv"

def live_analyze(pcap_path):
    if not os.path.exists(pcap_path):
        print(f"Error: File {pcap_path} not found.")
        return

    # 1. Feature Extraction
    if os.path.exists(TEMP_CSV):
        os.remove(TEMP_CSV)

    try:
        # Running cicflowmeter with -c flag for CSV output
        subprocess.run(["cicflowmeter", "-f", pcap_path, "-c", TEMP_CSV], 
                       stdout=subprocess.DEVNULL, # Hide command output
                       stderr=subprocess.DEVNULL, 
                       check=True)
        time.sleep(0.5) 
    except Exception as e:
        print(f"Extraction Error: {e}")
        return

    if not os.path.exists(TEMP_CSV) or os.path.getsize(TEMP_CSV) == 0:
        print("Analysis Error: No flows extracted.")
        return

    # 2. Data Processing & Prediction
    try:
        df = pd.read_csv(TEMP_CSV)
        X_processed = clean_data(df)
        
        model = joblib.load(MODEL_PATH)
        
        # Define class mapping (Alphabetical order based on your training)
        class_names = ['Bot', 'BruteForce', 'DDoS', 'DoS', 'Infiltration', 'Misc', 'Normal', 'PortScan', 'Web']
        
        # Feature Alignment
        expected_features = None
        # --- IMPROVED FEATURE ALIGNMENT ---
        expected_features = None
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
        else:
            try:
                expected_features = model.get_booster().feature_names
            except:
                expected_features = None

        if expected_features is not None:
            # 1. Add missing columns with 0
            for col in expected_features:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            
            # 2. SELECT and REORDER (This fixes the 78 vs 77 mismatch)
            X_processed = X_processed[expected_features]
        else:
            # Fallback if model has no feature names: Force exact count
            print(f"[!] Warning: Model has no feature names. Forcing 77 features.")
            if X_processed.shape[1] > 77:
                X_processed = X_processed.iloc[:, :77] # Cut the extra column
            elif X_processed.shape[1] < 77:
                for i in range(X_processed.shape[1], 77):
                    X_processed[f'f{i}'] = 0
        # ----------------------------------

        # Prediction
        predictions = model.predict(X_processed.values)
        results = pd.Series(predictions).value_counts()
        
        # 3. Output Results
        print("\n" + "="*50)
        print(f" NIDS ANALYSIS REPORT | Source: {os.path.basename(pcap_path)}")
        print("="*50)
        print(f"{'TYPE':<15} | {'STATUS':<15} | {'FLOWS':<10}")
        print("-" * 50)
        
        for idx, count in results.items():
            label = class_names[int(idx)] if int(idx) < len(class_names) else f"ID_{idx}"
            status = "CLEAN" if label == "Normal" else "THREAT"
            print(f"{label:<15} | {status:<15} | {count:<10}")
            
        print("="*50 + "\n")

    except Exception as e:
        print(f"Prediction Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 live_analyze.py <path_to_pcap>")
    else:
        live_analyze(sys.argv[1])