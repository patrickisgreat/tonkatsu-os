#!/usr/bin/env python3
"""Rebuild preprocessed spectra and feature vectors in the SQLite database."""

import argparse
from pathlib import Path

from tonkatsu_os.database import RamanSpectralDatabase


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate preprocessed spectra and feature cache entries."
    )
    parser.add_argument("--limit", type=int, help="Optional limit of spectra to process")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("raman_spectra.db"),
        help="Path to the SQLite database file",
    )
    args = parser.parse_args()

    db = RamanSpectralDatabase(str(args.database))
    try:
        count = db.rebuild_feature_cache(limit=args.limit)
    finally:
        db.close()

    print(f"Rebuilt features for {count} spectra")


if __name__ == "__main__":
    main()
