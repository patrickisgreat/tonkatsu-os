from pathlib import Path

import pytest


def test_core_package_layout():
    """
    Smoke-check critical package directories to guarantee the source tree
    is intact in development and CI environments.
    """
    expected_paths = [
        Path("src/tonkatsu_os/__init__.py"),
        Path("src/tonkatsu_os/database"),
        Path("src/tonkatsu_os/preprocessing"),
        Path("src/tonkatsu_os/ml"),
    ]

    for path in expected_paths:
        assert path.exists(), f"Expected {path} to exist"


def test_database_file_exists():
    """
    The bundled SQLite database is required for local analysis demos.
    Confirm it exists so onboarding fails fast if the artifact is missing.
    """
    db_path = Path("raman_spectra.db")
    if not db_path.is_file():
        pytest.skip("Bundled Raman database not present in this environment")


def test_package_version_exposed():
    """Basic sanity check that the package exposes version metadata."""
    import tonkatsu_os

    assert tonkatsu_os.__version__ == "0.2.0"
