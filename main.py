import argparse

def main():
    parser = argparse.ArgumentParser(description="Network Intrusion Detection System")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", type=str, help="Path to a directory containing CICIDS2017 CSV files")
    mode.add_argument("--pcap", type=str, help="Path to a PCAP file for live/offline flow analysis")
    parser.add_argument("--model", type=str, default="xgb",
                        choices=["rf", "xgb", "lr", "knn"])
    parser.add_argument("--sample", type=int, default=None,
                        help="Optional number of rows to sample during training")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory where trained artifacts are saved")
    parser.add_argument("--no-imbalance", action="store_true",
                        help="Disable under-sampling, SMOTE, and XGBoost class weights during training")
    parser.add_argument("--signatures", action="store_true",
                        help="Enable supplemental PCAP signature alerts")

    args = parser.parse_args()

    if args.train:
        from src.pipeline.train_pipeline import run_train_pipeline

        run_train_pipeline(
            data_path=args.train,
            model_name=args.model,
            sample_size=args.sample,
            model_dir=args.model_dir if args.model_dir else None,
            use_imbalance=not args.no_imbalance
        )
    elif args.pcap:
        from live_analyze import live_analyze

        live_analyze(
            args.pcap,
            model_dir=args.model_dir if args.model_dir else None,
            use_signatures=args.signatures
        )

if __name__ == "__main__":
    main()
