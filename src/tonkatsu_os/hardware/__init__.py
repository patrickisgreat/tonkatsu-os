"""
Hardware interface modules for Raman spectrometer control.
"""

from .spectrometer import (
    BWTekSpectrometer,
    HardwareManager,
    SpectrometerAcquisitionError,
    SpectrometerConfig,
    SpectrometerConnectionError,
    SpectrometerError,
)

__all__ = [
    "BWTekSpectrometer",
    "HardwareManager",
    "SpectrometerConfig",
    "SpectrometerError",
    "SpectrometerConnectionError",
    "SpectrometerAcquisitionError",
]
