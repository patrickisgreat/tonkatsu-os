#!/usr/bin/env python3
"""List spectra and metadata from the local SQLite database."""

import argparse
import csv
import sqlite3
from pathlib import Path

import pandas as pd


def list_spectra(database: Path, limit: int) -> pd.DataFrame:
    """Return a DataFrame with the first `limit` spectra records."""
    conn = sqlite3.connect(database)
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


def export_spectra_csv(database: Path, output_path: Path) -> None:
    """Dump all spectra metadata to a CSV file."""
    conn = sqlite3.connect(database)
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
        writer.writerow(
            ["id", "compound_name", "chemical_formula", "cas_number", "acquisition_date"]
        )
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Tonkatsu spectra database.")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("raman_spectra.db"),
        help="Path to the SQLite database file.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of rows to display.")
    parser.add_argument(
        "--export",
        type=Path,
        help="Export the full spectra list to CSV instead of printing a preview.",
    )
    args = parser.parse_args()

    if args.export:
        export_spectra_csv(args.database, args.export)
        print(f"Exported spectra to {args.export}")
        return

    df = list_spectra(args.database, args.limit)
    if df.empty:
        print("No spectra found.")
        return

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
