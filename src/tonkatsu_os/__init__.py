"""
Tonkatsu-OS: AI-Powered Raman Molecular Identification Platform

A comprehensive, open-source Raman spectrometer control and AI-powered 
molecular identification platform.
"""

__version__ = "0.2.0"
__author__ = "Patrick"
__email__ = "patrick@tonkatsu-os.com"
__description__ = "AI-powered Raman spectroscopy molecular identification platform"

# Core imports for easy access
from tonkatsu_os.database.raman_database import RamanSpectralDatabase
from tonkatsu_os.preprocessing.advanced_preprocessor import AdvancedPreprocessor
from tonkatsu_os.ml.ensemble_classifier import EnsembleClassifier
from tonkatsu_os.core.spectrum_importer import SpectrumImporter
# from tonkatsu_os.visualization.spectral_visualizer import SpectralVisualizer  # Temporarily disabled due to matplotlib issue

__all__ = [
    "RamanSpectralDatabase",
    "AdvancedPreprocessor", 
    "EnsembleClassifier",
    "SpectrumImporter",
    # "SpectralVisualizer",  # Temporarily disabled
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]