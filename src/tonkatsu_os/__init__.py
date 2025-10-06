"""
Tonkatsu-OS: AI-Powered Raman Molecular Identification Platform

A comprehensive, open-source Raman spectrometer control and AI-powered 
molecular identification platform.
"""

from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"
__author__ = "Patrick"
__email__ = "patrick@tonkatsu-os.com"
__description__ = "AI-powered Raman spectroscopy molecular identification platform"

if TYPE_CHECKING:
    from tonkatsu_os.core.spectrum_importer import SpectrumImporter
    from tonkatsu_os.database.raman_database import RamanSpectralDatabase
    from tonkatsu_os.ml.ensemble_classifier import EnsembleClassifier
    from tonkatsu_os.preprocessing.advanced_preprocessor import AdvancedPreprocessor


def __getattr__(name: str) -> Any:
    """
    Lazily expose heavy-weight modules so importing tonkatsu_os does not require
    NumPy/SciPy to be ready. This keeps lightweight commands and smoke tests stable.
    """
    if name == "SpectrumImporter":
        from tonkatsu_os.core.spectrum_importer import SpectrumImporter

        return SpectrumImporter
    if name == "RamanSpectralDatabase":
        from tonkatsu_os.database.raman_database import RamanSpectralDatabase

        return RamanSpectralDatabase
    if name == "EnsembleClassifier":
        from tonkatsu_os.ml.ensemble_classifier import EnsembleClassifier

        return EnsembleClassifier
    if name == "AdvancedPreprocessor":
        from tonkatsu_os.preprocessing.advanced_preprocessor import AdvancedPreprocessor

        return AdvancedPreprocessor

    raise AttributeError(f"module 'tonkatsu_os' has no attribute {name!r}")


__all__ = [
    "RamanSpectralDatabase",
    "AdvancedPreprocessor",
    "EnsembleClassifier",
    "SpectrumImporter",
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]
