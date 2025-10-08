"""Utilities for training the Tonkatsu ensemble classifier."""

import json
import logging
import pickle
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from tonkatsu_os.database import RamanSpectralDatabase
from tonkatsu_os.ml.ensemble_classifier import EnsembleClassifier

logger = logging.getLogger(__name__)

MODEL_PATH = Path("trained_ensemble_model.pkl")
METRICS_PATH = Path("trained_ensemble_metrics.json")


def _prepare_dataset(
    db: RamanSpectralDatabase,
    min_samples_per_class: int = 3,
    rebuild_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    """Load feature vectors and labels from the database.

    Args:
        db: Open database instance.
        min_samples_per_class: Skip classes with fewer samples than this.
        rebuild_features: Whether to regenerate feature cache before loading.

    Returns:
        Tuple of (features, labels, kept_label_counts, excluded_label_counts).
    """
    if rebuild_features:
        logger.info("Rebuilding feature cache before training...")
        db.rebuild_feature_cache()

    cursor = db.conn.execute(
        """
        SELECT s.id, s.compound_name, sf.feature_vector
        FROM spectral_features sf
        JOIN spectra s ON sf.spectrum_id = s.id
        """
    )

    features: List[np.ndarray] = []
    labels: List[str] = []
    label_counts: Counter = Counter()

    for spectrum_id, compound_name, feature_blob in cursor.fetchall():
        if feature_blob is None:
            continue
        try:
            vec = pickle.loads(feature_blob)
        except Exception as exc:  # pragma: no cover - unexpected corruption
            logger.warning("Skipping spectrum %s due to feature decode error: %s", spectrum_id, exc)
            continue

        if not isinstance(vec, np.ndarray):
            vec = np.asarray(vec, dtype=float)

        if vec.size == 0:
            continue

        features.append(vec.astype(float))
        labels.append(compound_name)
        label_counts[compound_name] += 1

    if not features:
        raise ValueError("No spectral features found in database")

    features = np.vstack(features)
    labels = np.asarray(labels)

    excluded = {label: count for label, count in label_counts.items() if count < min_samples_per_class}
    if excluded:
        logger.info("Excluding %s labels with < %s samples", len(excluded), min_samples_per_class)

    mask = np.array([label_counts[label] >= min_samples_per_class for label in labels])
    X_filtered = features[mask]
    y_filtered = labels[mask]

    kept_counts = Counter(y_filtered)
    if len(kept_counts) < 2:
        raise ValueError("Not enough classes with sufficient samples for training")

    return X_filtered, y_filtered, dict(kept_counts), excluded


def _adjust_validation_split(validation_split: float, label_counts: Dict[str, int]) -> float:
    min_count = min(label_counts.values())
    min_fraction = 1.0 / float(min_count)
    adjusted = max(validation_split, min_fraction)
    if adjusted >= 1.0:
        adjusted = min_fraction if min_count > 1 else 0.5
    if adjusted >= 1.0:
        raise ValueError("Validation split too large for available samples")
    return adjusted


def train_ensemble_model(
    db_path: Path = Path("raman_spectra.db"),
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    use_pca: bool = True,
    use_pls: bool = False,
    n_components: int = 50,
    validation_split: float = 0.2,
    min_samples_per_class: int = 2,
    rebuild_features: bool = False,
) -> Dict:
    """Train the ensemble classifier and persist model + metrics."""
    start_time = time.time()
    db = RamanSpectralDatabase(str(db_path))
    try:
        X, y, kept_counts, excluded = _prepare_dataset(
            db, min_samples_per_class=min_samples_per_class, rebuild_features=rebuild_features
        )
    finally:
        db.close()

    adjusted_split = _adjust_validation_split(validation_split, kept_counts)

    # Ensure PCA component count is valid
    actual_components = n_components
    if use_pca:
        max_components = min(X.shape[1], X.shape[0] - 1)
        if max_components < 1:
            use_pca = False
            actual_components = 0
        else:
            actual_components = min(n_components, max_components)
            if actual_components < 1:
                use_pca = False

    classifier = EnsembleClassifier(
        use_pca=use_pca,
        n_components=(actual_components or 1),
        use_pls=use_pls,
    )
    metrics = classifier.train(X, y, validation_split=adjusted_split)
    training_time = time.time() - start_time

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    classifier.save_model(str(model_path))

    metrics_out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "training_time": training_time,
        "use_pca": use_pca,
        "use_pls": use_pls,
        "n_components": actual_components if use_pca else None,
        "validation_split": adjusted_split,
        "min_samples_per_class": min_samples_per_class,
        "n_samples": int(X.shape[0]),
        "labels": classifier.class_names.tolist(),
        "label_counts": {label: int(count) for label, count in kept_counts.items()},
        "excluded_labels": {label: int(count) for label, count in excluded.items()},
        **metrics,
    }

    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    logger.info("Training finished in %.2fs. Model saved to %s", training_time, model_path)
    return metrics_out
