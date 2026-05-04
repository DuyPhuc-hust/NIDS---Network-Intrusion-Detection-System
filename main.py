import argparse
import joblib
from src.pipeline.train_pipeline import run_train_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str)

    args = parser.parse_args()

    model, scaler, encoder, label_encoder = run_train_pipeline(args.train)

    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoder, "models/encoder.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    print("[+] Model saved!")


if __name__ == "__main__":
    main()