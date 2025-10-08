#!/usr/bin/env python3
<<<<<<< HEAD
"""Rebuild the spectral feature cache in the SQLite database."""
=======
"""Rebuild preprocessed spectra and feature vectors in the SQLite database."""
>>>>>>> main

import argparse
from pathlib import Path

from tonkatsu_os.database import RamanSpectralDatabase


def main():
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="Regenerate preprocessed spectra and features")
    parser.add_argument("--database", type=Path, default=Path("raman_spectra.db"))
    parser.add_argument("--limit", type=int, help="Optional limit of spectra to process")
=======
    parser = argparse.ArgumentParser(description="Regenerate spectral features cache")
    parser.add_argument("--limit", type=int, help="Optional limit of spectra to process")
    parser.add_argument(
        "--database", type=Path, default=Path("raman_spectra.db"), help="Path to SQLite DB"
    )
>>>>>>> main
    args = parser.parse_args()

    db = RamanSpectralDatabase(str(args.database))
    try:
        count = db.rebuild_feature_cache(limit=args.limit)
    finally:
        db.close()

    print(f"Rebuilt features for {count} spectra")


if __name__ == "__main__":
    main()
