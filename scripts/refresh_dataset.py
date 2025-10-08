#!/usr/bin/env python3
"""Convenience script to ingest curated datasets and rebuild features."""

import argparse
from pathlib import Path

from tonkatsu_os.database import RamanSpectralDatabase

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
        db.rebuild_feature_cache()
    finally:
        db.close()

    print(f"Imported {len(spectra)} spectra and rebuilt feature cache")


if __name__ == "__main__":
    main()
