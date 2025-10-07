#!/usr/bin/env python3
"""Quick CLI to analyze a spectrum using the local FastAPI app."""

import argparse
import json
from pathlib import Path

import asyncio
import numpy as np
import httpx

from tonkatsu_os.api.main import app


async def analyze(spectrum: np.ndarray, preprocess: bool = True):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://local") as client:
        payload = {
            "spectrum_data": spectrum.astype(float).tolist(),
            "preprocess": preprocess,
        }
        response = await client.post("/api/analysis/analyze", json=payload)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Analyze a spectrum via the Tonkatsu API")
    parser.add_argument("spectrum", help="Path to a JSON or CSV file with spectrum data")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable preprocessing")
    args = parser.parse_args()

    path = Path(args.spectrum)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        spectrum = np.array(data["spectrum_data"] if isinstance(data, dict) else data)
    else:
        spectrum = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)

    result = asyncio.run(analyze(spectrum, preprocess=not args.no_preprocess))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

