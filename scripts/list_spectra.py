#!/usr/bin/env python3
<<<<<<< HEAD
"""List spectra metadata from the local SQLite database."""

import argparse
=======
"""List spectra and metadata from the bundled SQLite database."""

import argparse
import csv
>>>>>>> main
import sqlite3
from pathlib import Path

import pandas as pd

<<<<<<< HEAD

def list_spectra(database: Path, limit: int) -> pd.DataFrame:
    conn = sqlite3.connect(database)
    try:
        query = (
            "SELECT id, compound_name, chemical_formula, cas_number, acquisition_date "
            "FROM spectra ORDER BY id ASC LIMIT ?"
        )
        return pd.read_sql_query(query, conn, params=(limit,))
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="List spectra entries")
    parser.add_argument("--database", type=Path, default=Path("raman_spectra.db"))
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    df = list_spectra(args.database, args.limit)
    if df.empty:
        print("No spectra found.")
    else:
        print(df.to_string(index=False))
=======
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
>>>>>>> main


if __name__ == "__main__":
    main()
