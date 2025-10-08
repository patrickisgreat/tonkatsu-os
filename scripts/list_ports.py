#!/usr/bin/env python3
"""
List available serial ports for Tonkatsu-OS spectrometer diagnostics.

This helper wraps ``serial.tools.list_ports`` and prints a concise
summary (or JSON output) of every device that pyserial can detect.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List


def _load_pyserial():
    try:
        from serial.tools import list_ports  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime environment
        print(
            "pyserial is not installed. Install it with `poetry install` or "
            "`pip install pyserial` to use this tool.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return list_ports


def _serialize_port(port) -> Dict[str, Any]:
    return {
        "device": port.device,
        "description": port.description,
        "hwid": port.hwid,
        "manufacturer": getattr(port, "manufacturer", None),
        "product": getattr(port, "product", None),
        "serial_number": getattr(port, "serial_number", None),
        "location": getattr(port, "location", None),
    }


def list_serial_ports() -> List[Dict[str, Any]]:
    list_ports = _load_pyserial()
    return [_serialize_port(port) for port in list_ports.comports()]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show extended metadata for each port.",
    )
    args = parser.parse_args(argv)

    ports = list_serial_ports()

    if args.json:
        print(json.dumps(ports, indent=2))
        return 0

    if not ports:
        print("No serial ports detected.")
        return 0

    for index, info in enumerate(ports, start=1):
        header = f"[{index}] {info['device']} — {info['description']}"
        print(header)
        print(f"    HWID: {info['hwid']}")
        if args.verbose:
            if info.get("manufacturer"):
                print(f"    Manufacturer: {info['manufacturer']}")
            if info.get("product"):
                print(f"    Product: {info['product']}")
            if info.get("serial_number"):
                print(f"    Serial: {info['serial_number']}")
            if info.get("location"):
                print(f"    Location: {info['location']}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
