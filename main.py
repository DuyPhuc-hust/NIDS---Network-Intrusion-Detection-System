import argparse
from src.pipeline.train_pipeline import run_train_pipeline
from src.utils.helpers import save_artifacts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="Path to dataset folder")
    parser.add_argument("--sample", type=int, default=None)

    args = parser.parse_args()

    if args.train:
        model, scaler, features = run_train_pipeline(
            args.train,
            sample_size=args.sample
        )

        save_artifacts(model, scaler, features)


if __name__ == "__main__":
    main()