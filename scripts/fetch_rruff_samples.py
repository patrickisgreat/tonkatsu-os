#!/usr/bin/env python3
"""Fetch a deterministic subset of RRUFF spectra defined in samples.yaml."""

import argparse
import logging
import zipfile
from pathlib import Path
from typing import Dict, List

import yaml

from tonkatsu_os.core.data_loader import RRUFFDataLoader
from tonkatsu_os.database import RamanSpectralDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fetch_rruff_samples")

SAMPLES_CONFIG = Path("data/raw/rruff/samples.yaml")


def fetch_samples(output_dir: Path) -> List[Dict]:
    """Load curated RRUFF spectra from cached zip archives."""
    if not SAMPLES_CONFIG.exists():
        raise FileNotFoundError(
            f"Missing {SAMPLES_CONFIG}. Please provide curated sample metadata."
        )

    loader = RRUFFDataLoader(data_dir=str(output_dir))
    config = yaml.safe_load(SAMPLES_CONFIG.read_text())

    spectra: List[Dict] = []
    for entry in config.get("samples", []):
        filenames = set(entry.get("filenames", []))
        local_zip = entry.get("local_zip")
        if not local_zip:
            raise ValueError("Each samples.yaml entry must include 'local_zip'.")

        zip_path = Path(local_zip)
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing cached RRUFF zip: {zip_path}")

        logger.info("Reading %s", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name not in filenames:
                    continue

                logger.info("Processing %s", name)
                with zf.open(name) as file_obj:
                    content = file_obj.read().decode("utf-8", errors="ignore")
                parsed = loader._parse_rruff_spectrum(content)
                if parsed is None or len(parsed) == 0:
                    continue

                parts = name.split("__")
                mineral = parts[0].replace("_", " ")
                rruff_id = parts[1] if len(parts) > 1 else ""
                is_infrared = "Infrared" in parts

                spectra.append(
                    {
                        "compound_name": mineral,
                        "chemical_formula": "",
                        "spectrum_data": parsed,
                        "measurement_conditions": (
                            "RRUFF infrared curated sample"
                            if is_infrared
                            else "RRUFF raman curated sample"
                        ),
                        "laser_wavelength": 785.0 if is_infrared else 532.0,
                        "metadata": {
                            "source": "RRUFF curated",
                            "original_filename": name,
                            "zip_path": str(zip_path),
                            "rruff_id": rruff_id,
                        },
                    }
                )

    logger.info("Collected %s spectra", len(spectra))
    return spectra


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch curated RRUFF samples.")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("raman_spectra.db"),
        help="Path to the SQLite database file.",
    )
    parser.add_argument("--limit", type=int, help="Limit spectra added to database")
    args = parser.parse_args()

    output_dir = Path("data/raw/rruff")
    output_dir.mkdir(parents=True, exist_ok=True)

    spectra = fetch_samples(output_dir)
    if args.limit is not None:
        spectra = spectra[: args.limit]

    logger.info("Integrating %s spectra", len(spectra))

    db = RamanSpectralDatabase(str(args.database))
    try:
        for spec in spectra:
            db.add_spectrum(**spec)
    finally:
        db.close()

    logger.info("Imported %s curated RRUFF spectra", len(spectra))


if __name__ == "__main__":
    main()
