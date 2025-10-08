#!/usr/bin/env python3
<<<<<<< HEAD
"""Identify or prune duplicate spectra based on spectral hash."""
=======
"""Print potential duplicate spectra based on spectral hash."""
>>>>>>> main

import argparse
from pathlib import Path

from tonkatsu_os.database import RamanSpectralDatabase


def main():
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="List or prune duplicate spectra")
    parser.add_argument("--database", type=Path, default=Path("raman_spectra.db"))
    parser.add_argument("--prune", action="store_true", help="Remove lower-ranked duplicates")
=======
    parser = argparse.ArgumentParser(description="List duplicate spectra by spectral hash")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("raman_spectra.db"),
        help="Path to SQLite database",
    )
>>>>>>> main
    args = parser.parse_args()

    db = RamanSpectralDatabase(str(args.database))
    try:
<<<<<<< HEAD
        if args.prune:
            removed = db.remove_duplicate_spectra()
            duplicates = db.find_duplicates()
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
    else:
        print(f"Found {len(duplicates)} duplicate groups:")
        for group in duplicates:
            print(f"Hash {group['hash']}: spectra {group['spectrum_ids']}")
=======
        duplicates = db.find_duplicates()
    finally:
        db.close()

    if not duplicates:
        print("No duplicates detected.")
        return

    print(f"Found {len(duplicates)} duplicate groups:")
    for group in duplicates:
        print(f"Hash {group['hash']}: spectra {group['spectrum_ids']}")
>>>>>>> main


if __name__ == "__main__":
    main()
