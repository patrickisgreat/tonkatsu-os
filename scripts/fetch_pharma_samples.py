#!/usr/bin/env python3
"""Import curated pharmaceutical Raman spectra from cached archives."""

import argparse
import logging
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from tonkatsu_os.database import RamanSpectralDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fetch_pharma_samples")

SAMPLES_CONFIG = Path("data/raw/pharma/samples.yaml")


def _load_config() -> Dict:
    if not SAMPLES_CONFIG.exists():
        raise FileNotFoundError(
            f"Missing {SAMPLES_CONFIG}. Please create it with dataset path and labels."
        )
    return yaml.safe_load(SAMPLES_CONFIG.read_text())


def fetch_samples() -> List[Dict]:
    config = _load_config()
    dataset_zip = Path(config["dataset_zip"])
    if not dataset_zip.exists():
        raise FileNotFoundError(f"Cached dataset not found: {dataset_zip}")

    targets = {entry["label"]: entry["count"] for entry in config.get("samples", [])}
    remaining = targets.copy()
    collected: Dict[str, List[Dict]] = defaultdict(list)

    with zipfile.ZipFile(dataset_zip, "r") as zf:
        with zf.open("raman_spectra_api_compounds.csv") as csv_file:
            reader = pd.read_csv(csv_file, chunksize=512)
            row_offset = 0
            for chunk in reader:
                if all(remaining[label] <= 0 for label in remaining):
                    break

                label_series = chunk["label"]
                intensity_cols = chunk.columns.drop("label")

                for label, count in list(remaining.items()):
                    if count <= 0:
                        continue
                    matches = chunk[label_series == label]
                    if matches.empty:
                        continue

                    take = min(count, len(matches))
                    for idx, (_, row) in enumerate(matches.iloc[:take].iterrows()):
                        intensities = row[intensity_cols].astype(float).to_numpy()
                        entry = {
                            "compound_name": label,
                            "chemical_formula": "",
                            "spectrum_data": intensities,
                            "measurement_conditions": "Pharma API curated sample",
                            "laser_wavelength": 785.0,
                            "metadata": {
                                "source": "Pharma curated",
                                "dataset_zip": str(dataset_zip),
                                "row_index": int(row_offset + row.name),
                            },
                        }
                        collected[label].append(entry)
                    remaining[label] -= take

                row_offset += len(chunk)

    spectra: List[Dict] = []
    for label, entries in collected.items():
        logger.info("Collected %s spectra for %s", len(entries), label)
        spectra.extend(entries)

    missing = [label for label, count in remaining.items() if count > 0]
    if missing:
        logger.warning("Did not fulfill counts for labels: %s", missing)

    return spectra


def main():
    parser = argparse.ArgumentParser(description="Fetch curated Pharma spectra")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("raman_spectra.db"),
        help="Path to SQLite database",
    )
    parser.add_argument("--limit", type=int, help="Optional limit for inserted spectra")
    args = parser.parse_args()

    spectra = fetch_samples()
    if args.limit is not None:
        spectra = spectra[: args.limit]

    db = RamanSpectralDatabase(str(args.database))
    try:
        for spec in spectra:
            db.add_spectrum(**spec)
    finally:
        db.close()

    logger.info("Imported %s Pharma spectra", len(spectra))


if __name__ == "__main__":
    main()
