#!/usr/bin/env python3
"""List spectra and metadata from the bundled SQLite database."""

import argparse
import csv
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("raman_spectra.db")


def list_spectra(limit: int = 20):
    """Return a Pandas DataFrame with the first `limit` spectra records."""
    conn = sqlite3.connect(DB_PATH)
    try:
        query = """
        SELECT id, compound_name, chemical_formula, cas_number, acquisition_date
        FROM spectra
        ORDER BY id ASC
        LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
    finally:
        conn.close()
    return df


def export_spectra_csv(output_path: Path):
    """Dump all spectra metadata to a CSV file."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, compound_name, chemical_formula, cas_number, acquisition_date FROM spectra"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "compound_name", "chemical_formula", "cas_number", "acquisition_date"])
        writer.writerows(rows)


def main():
    """CLI entry point for inspecting or exporting the spectra database."""
    parser = argparse.ArgumentParser(description="Inspect Tonkatsu spectra database.")
    parser.add_argument("--limit", type=int, default=20, help="Number of rows to display")
    parser.add_argument("--export", type=Path, help="Export the full spectra list to CSV")
    args = parser.parse_args()

    if args.export:
        export_spectra_csv(args.export)
        print(f"Exported spectra to {args.export}")
        return

    df = list_spectra(limit=args.limit)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
