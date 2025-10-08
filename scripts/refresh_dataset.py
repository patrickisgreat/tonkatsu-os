#!/usr/bin/env python3
<<<<<<< HEAD
"""Convenience script to ingest curated datasets and rebuild features."""
=======
"""Convenience script to import curated data and rebuild feature cache."""
>>>>>>> main

import argparse
from pathlib import Path

from tonkatsu_os.database import RamanSpectralDatabase

<<<<<<< HEAD
from fetch_pharma_samples import fetch_samples as fetch_pharma_samples
from fetch_rruff_samples import fetch_samples as fetch_rruff_samples


def main():
    parser = argparse.ArgumentParser(description="Refresh curated spectral datasets")
    parser.add_argument("--database", type=Path, default=Path("raman_spectra.db"))
    parser.add_argument("--limit", type=int, help="Optional combined spectrum limit")
    args = parser.parse_args()

    rruff = fetch_rruff_samples(Path("data/raw/rruff"))
    pharma = fetch_pharma_samples()
    spectra = rruff + pharma
    if args.limit is not None:
        spectra = spectra[: args.limit]

    db = RamanSpectralDatabase(str(args.database))
    try:
        for spec in spectra:
            db.add_spectrum(**spec)
=======
from fetch_rruff_samples import fetch_samples as fetch_rruff_samples
from fetch_pharma_samples import fetch_samples as fetch_pharma_samples


def main():
    parser = argparse.ArgumentParser(description="Refresh local spectral dataset")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("raman_spectra.db"),
        help="Path to SQLite database",
    )
    parser.add_argument("--limit", type=int, help="Limit number of spectra to ingest")
    args = parser.parse_args()

    db_path = args.database
    db = RamanSpectralDatabase(str(db_path))
    try:
        rruff_spectra = fetch_rruff_samples(Path("data/raw/rruff"))
        pharma_spectra = fetch_pharma_samples()

        spectra = rruff_spectra + pharma_spectra
        if args.limit is not None:
            spectra = spectra[: args.limit]

        for spec in spectra:
            db.add_spectrum(**spec)

>>>>>>> main
        db.rebuild_feature_cache()
    finally:
        db.close()

<<<<<<< HEAD
    print(f"Imported {len(spectra)} spectra and rebuilt feature cache")
=======
    print(
        f"Imported {len(spectra)} spectra "
        f"(RRUFF {len(rruff_spectra)}, Pharma {len(pharma_spectra)}) "
        "and rebuilt feature cache"
    )
>>>>>>> main


if __name__ == "__main__":
    main()
