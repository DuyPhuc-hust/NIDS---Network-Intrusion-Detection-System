import argparse
from src.pipeline.train_pipeline import run_train_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--model", type=str, default="xgb",
                        choices=["rf", "xgb", "lr", "knn"])
    parser.add_argument("--sample", type=int, default=None)

    args = parser.parse_args()

    run_train_pipeline(
        data_path=args.train,
        model_name=args.model,
        sample_size=args.sample
    )

if __name__ == "__main__":
    main()