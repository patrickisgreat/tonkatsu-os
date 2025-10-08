#!/usr/bin/env python3
"""Import curated pharmaceutical spectra from cached dataset."""

import argparse
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from tonkatsu_os.database import RamanSpectralDatabase

SAMPLES_CONFIG = Path("data/raw/pharma/samples.yaml")


def fetch_samples() -> List[Dict]:
    config = yaml.safe_load(SAMPLES_CONFIG.read_text())
    dataset_zip = Path(config["dataset_zip"])
    if not dataset_zip.exists():
        raise FileNotFoundError(f"Cached dataset not found: {dataset_zip}")

    targets = {entry["label"]: entry["count"] for entry in config.get("samples", [])}
    remaining = targets.copy()
    collected: Dict[str, List[Dict]] = defaultdict(list)

    with zipfile.ZipFile(dataset_zip, "r") as zf:
        with zf.open("raman_spectra_api_compounds.csv") as csv_file:
            reader = pd.read_csv(csv_file, chunksize=512)
            offset = 0
            for chunk in reader:
                if all(remaining[label] <= 0 for label in remaining):
                    break

                intensity_cols = chunk.columns.drop("label")
                for label, need in list(remaining.items()):
                    if need <= 0:
                        continue
                    matches = chunk[chunk["label"] == label]
                    if matches.empty:
                        continue
                    take = min(need, len(matches))
                    for _, row in matches.iloc[:take].iterrows():
                        intensities = row[intensity_cols].astype(float).to_numpy()
                        collected[label].append(
                            {
                                "compound_name": label,
                                "chemical_formula": "",
                                "spectrum_data": intensities,
                                "measurement_conditions": "Pharma API curated sample",
                                "laser_wavelength": 785.0,
                                "metadata": {
                                    "source": "Pharma curated",
                                    "dataset_zip": str(dataset_zip),
                                    "row_index": int(offset + row.name),
                                },
                            }
                        )
                    remaining[label] -= take
                offset += len(chunk)

    spectra: List[Dict] = []
    for label, entries in collected.items():
        spectra.extend(entries)
    return spectra


def main():
    parser = argparse.ArgumentParser(description="Import curated pharma spectra")
    parser.add_argument("--database", type=Path, default=Path("raman_spectra.db"))
    parser.add_argument("--limit", type=int, help="Limit number of spectra")
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

    print(f"Imported {len(spectra)} Pharma spectra")


if __name__ == "__main__":
    main()
