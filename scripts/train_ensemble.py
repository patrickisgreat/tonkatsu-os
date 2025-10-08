#!/usr/bin/env python3
"""Train the Tonkatsu ensemble classifier from the command line."""

import argparse
from pathlib import Path

from tonkatsu_os.ml.training_pipeline import train_ensemble_model, MODEL_PATH, METRICS_PATH


def main():
    parser = argparse.ArgumentParser(description="Train ensemble model on the local database")
    parser.add_argument("--database", type=Path, default=Path("raman_spectra.db"), help="Path to SQLite DB")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH, help="Output path for trained model")
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH, help="Output path for metrics JSON")
    parser.add_argument("--no-pca", action="store_true", help="Disable PCA before training")
    parser.add_argument("--use-pls", action="store_true", help="Include PLS regression in ensemble")
    parser.add_argument("--n-components", type=int, default=50, help="Number of PCA components if enabled")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--min-samples", type=int, default=3, help="Minimum samples per class to include")
    parser.add_argument("--rebuild-features", action="store_true", help="Recompute feature cache before training")
    args = parser.parse_args()

    metrics = train_ensemble_model(
        db_path=args.database,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        use_pca=not args.no_pca,
        use_pls=args.use_pls,
        n_components=args.n_components,
        validation_split=args.validation_split,
        min_samples_per_class=args.min_samples,
        rebuild_features=args.rebuild_features,
    )

    print("Training complete. Metrics written to", args.metrics_path)
    print("Ensemble accuracy:", metrics["ensemble_accuracy"])
if __name__ == "__main__":
    main()
