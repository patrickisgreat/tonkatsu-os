#!/usr/bin/env python3
"""
Replay a recorded spectrum through the Tonkatsu-OS acquisition simulator.

Given a CSV or JSON file containing spectral intensities, this script can:
  * Validate and summarize the data locally.
  * Optionally send the spectrum to the FastAPI backend using the simulator
    pathway so that `/api/acquisition/acquire` returns deterministic data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

try:
    import requests
except ImportError as exc:  # pragma: no cover - runtime environment
    print("The `requests` package is required. Install it with `poetry install`.", file=sys.stderr)
    raise SystemExit(1) from exc


def _normalize_values(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    if data.ndim > 1:
        data = data.flatten()
    if data.size == 0:
        raise ValueError("Spectrum data is empty")
    return data


def _load_json(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        candidates: Iterable = payload.get("spectrum_data") or payload.get("data") or ()
    else:
        candidates = payload
    return _normalize_values(list(candidates))


def _load_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return _normalize_values(data)


def load_spectrum(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    if suffix in {".csv", ".txt"}:
        return _load_csv(path)
    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")


def summarize_spectrum(data: np.ndarray) -> str:
    return (
        f"points={data.size}, min={float(np.min(data)):.2f}, "
        f"max={float(np.max(data)):.2f}, mean={float(np.mean(data)):.2f}, "
        f"std={float(np.std(data)):.2f}"
    )


def send_to_backend(
    spectrum_path: Path,
    integration_time: float,
    api_base: str,
) -> None:
    base = api_base.rstrip("/")
    simulation_file = spectrum_path.resolve().as_posix()

    connect_params = {
        "simulate": "true",
        "simulation_file": simulation_file,
        "port": "simulator",
    }

    print(f"→ Initializing simulator via {base}/acquisition/connect")
    response = requests.post(f"{base}/acquisition/connect", params=connect_params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    print(f"  connect: success={payload.get('success')} message={payload.get('message')}")

    acquire_payload = {
        "integration_time": integration_time,
        "simulate": True,
        "simulation_file": simulation_file,
    }

    print(f"→ Requesting spectrum via {base}/acquisition/acquire")
    acquire_resp = requests.post(f"{base}/acquisition/acquire", json=acquire_payload, timeout=30)
    acquire_resp.raise_for_status()
    acquisition = acquire_resp.json()

    data = acquisition.get("data", [])
    print(
        f"  acquire: source={acquisition.get('source')}, "
        f"points={len(data)}, acquired_at={acquisition.get('acquired_at')}"
    )
    if data:
        preview = ", ".join(f"{float(val):.1f}" for val in data[:10])
        print(f"  sample: {preview}{' ...' if len(data) > 10 else ''}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="Path to recorded spectrum (CSV or JSON).")
    parser.add_argument(
        "--integration-time",
        type=float,
        default=200.0,
        help="Integration time in milliseconds to report to the backend (default: 200).",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/api",
        help="Base URL for the Tonkatsu-OS API.",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Send the spectrum to the backend simulator after validation.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    spectrum_path: Path = args.file.expanduser()
    if not spectrum_path.exists():
        print(f"File not found: {spectrum_path}", file=sys.stderr)
        return 1

    try:
        spectrum = load_spectrum(spectrum_path)
    except Exception as exc:
        print(f"Failed to load spectrum: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded spectrum from {spectrum_path.as_posix()}")
    print("  " + summarize_spectrum(spectrum))

    preview = ", ".join(f"{float(val):.1f}" for val in spectrum[:10])
    print(f"  sample: {preview}{' ...' if spectrum.size > 10 else ''}")

    if args.send:
        try:
            send_to_backend(spectrum_path, args.integration_time, args.api_url)
        except requests.RequestException as exc:  # pragma: no cover - network operations
            print(f"Failed to contact backend: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
