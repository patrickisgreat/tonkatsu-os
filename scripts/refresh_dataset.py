#!/usr/bin/env python3
"""Convenience script to import curated data and rebuild feature cache."""

import argparse
from pathlib import Path

from tonkatsu_os.database import RamanSpectralDatabase

from fetch_rruff_samples import fetch_samples


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
        spectra = fetch_samples(Path("data/raw/rruff"))
        if args.limit is not None:
            spectra = spectra[: args.limit]

        for spec in spectra:
            db.add_spectrum(**spec)

        db.rebuild_feature_cache()
    finally:
        db.close()

    print(f"Imported {len(spectra)} spectra and rebuilt feature cache")


if __name__ == "__main__":
    main()
