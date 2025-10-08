#!/usr/bin/env python3
"""List spectra metadata from the local SQLite database."""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


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


if __name__ == "__main__":
    main()
