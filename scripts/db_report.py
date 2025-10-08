#!/usr/bin/env python3
"""Generate summary statistics for the Raman spectra database."""

import argparse
import ast
import sqlite3
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Report database statistics")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("raman_spectra.db"),
        help="Path to SQLite database",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.database)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM spectra")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT compound_name) FROM spectra")
    unique_compounds = cur.fetchone()[0]

    cur.execute("SELECT metadata FROM spectra")
    sources_counter = Counter()
    for (metadata_str,) in cur.fetchall():
        source = "unknown"
        if metadata_str:
            try:
                metadata = ast.literal_eval(metadata_str)
                source = metadata.get("source", source)
            except (ValueError, SyntaxError):
                pass
        sources_counter[source] += 1

    cur.execute(
        """
        SELECT spectral_hash, COUNT(*)
        FROM spectral_features
        GROUP BY spectral_hash
        HAVING COUNT(*) > 1
        """
    )
    duplicates = cur.fetchall()
    conn.close()

    print(f"Total spectra: {total}")
    print(f"Unique compounds: {unique_compounds}")
    print("By source:")
    for source, count in sources_counter.most_common():
        print(f"  {source}: {count}")
    print(f"Duplicate groups: {len(duplicates)}")


if __name__ == "__main__":
    main()
