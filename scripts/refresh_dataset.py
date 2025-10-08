#!/usr/bin/env python3
"""Convenience script to import curated data and rebuild the feature cache."""

import argparse
from pathlib import Path

from tonkatsu_os.database import RamanSpectralDatabase

from fetch_rruff_samples import fetch_samples as fetch_rruff_samples
from fetch_pharma_samples import fetch_samples as fetch_pharma_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh curated spectral datasets.")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("raman_spectra.db"),
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional combined spectrum limit to ingest before rebuilding features.",
    )
    args = parser.parse_args()

    rruff_spectra = fetch_rruff_samples(Path("data/raw/rruff"))
    pharma_spectra = fetch_pharma_samples()

    spectra = rruff_spectra + pharma_spectra
    if args.limit is not None:
        spectra = spectra[: args.limit]

    db = RamanSpectralDatabase(str(args.database))
    try:
        for spec in spectra:
            db.add_spectrum(**spec)
        db.rebuild_feature_cache()
    finally:
        db.close()

    print(
        f"Imported {len(spectra)} spectra "
        f"(RRUFF {len(rruff_spectra)}, Pharma {len(pharma_spectra)}) "
        "and rebuilt feature cache"
    )


if __name__ == "__main__":
    main()
