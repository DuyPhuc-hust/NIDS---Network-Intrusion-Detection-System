import argparse
from src.models.train import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="Path to training CSV")

    args = parser.parse_args()

    if args.train:
        train(args.train)

if __name__ == "__main__":
    main()