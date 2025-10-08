#!/usr/bin/env python3
"""Identify or prune duplicate spectra based on spectral hash."""

import argparse
from pathlib import Path

from tonkatsu_os.database import RamanSpectralDatabase


def main() -> None:
    parser = argparse.ArgumentParser(description="List or prune duplicate spectra.")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("raman_spectra.db"),
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Remove lower-ranked duplicates before reporting results.",
    )
    args = parser.parse_args()

    db = RamanSpectralDatabase(str(args.database))
    try:
        if args.prune:
            removed = db.remove_duplicate_spectra()
        else:
            removed = []
        duplicates = db.find_duplicates()
    finally:
        db.close()

    if args.prune:
        if removed:
            print(f"Removed {len(removed)} spectra: {removed}")
        else:
            print("No duplicates removed.")

    if not duplicates:
        print("No duplicates detected.")
        return

    print(f"Found {len(duplicates)} duplicate groups:")
    for group in duplicates:
        print(f"Hash {group['hash']}: spectra {group['spectrum_ids']}")


if __name__ == "__main__":
    main()
