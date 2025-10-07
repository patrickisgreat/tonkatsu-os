#!/usr/bin/env python3
"""Print potential duplicate spectra based on spectral hash."""

import argparse
from pathlib import Path

from tonkatsu_os.database import RamanSpectralDatabase


def main():
    parser = argparse.ArgumentParser(description="List duplicate spectra by spectral hash")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("raman_spectra.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Remove lower-quality duplicates, keeping the best spectrum per hash",
    )
    args = parser.parse_args()

    db = RamanSpectralDatabase(str(args.database))
    try:
        if args.prune:
            removed = db.remove_duplicate_spectra()
            duplicates = db.find_duplicates()
        else:
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
