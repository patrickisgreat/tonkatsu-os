"""
Hardware acquisition API routes for spectrometer control.
"""

import logging
import time
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException

from tonkatsu_os.hardware import HardwareManager

from ..models import AcquisitionRequest, ApiResponse, HardwareStatus

router = APIRouter()
logger = logging.getLogger(__name__)

# Global hardware manager instance
hardware_manager = HardwareManager()


@router.post("/acquire", response_model=list)
async def acquire_spectrum(request: AcquisitionRequest):
    """
    Acquire a spectrum from the connected spectrometer.

    If no hardware is connected, this generates a synthetic spectrum
    for demonstration purposes.
    """
    try:
        logger.info(f"Acquiring spectrum with integration time: {request.integration_time}ms")

        # Try to acquire from real hardware first
        spectrum = hardware_manager.acquire_spectrum(request.integration_time)

        if spectrum is not None:
            logger.info("Acquired spectrum from hardware")
            return spectrum.tolist()
        else:
            # Fall back to synthetic spectrum for demo
            logger.info("No hardware connected, generating synthetic spectrum")
            spectrum = _generate_demo_spectrum(request.integration_time)
            return spectrum.tolist()

    except Exception as e:
        logger.error(f"Error acquiring spectrum: {e}")
        raise HTTPException(status_code=500, detail=f"Acquisition failed: {str(e)}")


@router.get("/status", response_model=HardwareStatus)
async def get_hardware_status():
    """Get current hardware connection status."""
    try:
        # Get status from hardware manager
        status = hardware_manager.get_spectrometer_status()

        # Convert to expected format
        hardware_status = {
            "connected": status["connected"],
            "port": status["port"],
            "laser_status": "ready" if status["connected"] else "disconnected",
            "temperature": status.get("temperature", 25.0),
            "last_communication": status.get("last_communication"),
        }

        return HardwareStatus(**hardware_status)

    except Exception as e:
        logger.error(f"Error getting hardware status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/connect", response_model=ApiResponse)
async def connect_hardware(port: str = "/dev/ttyUSB0"):
    """Connect to spectrometer hardware."""
    try:
        logger.info(f"Attempting to connect to B&W Tek spectrometer on port: {port}")

        # Attempt real hardware connection
        success = hardware_manager.connect_spectrometer(port)

        if success:
            return ApiResponse(success=True, message=f"Connected to B&W Tek spectrometer on {port}")
        else:
            return ApiResponse(
                success=False,
                message=f"Failed to connect to spectrometer on {port}. Check if device is connected and port is correct.",
            )

    except Exception as e:
        logger.error(f"Error connecting to hardware: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disconnect", response_model=ApiResponse)
async def disconnect_hardware():
    """Disconnect from spectrometer hardware."""
    try:
        status = hardware_manager.get_spectrometer_status()

        if status["connected"]:
            success = hardware_manager.disconnect_spectrometer()

            if success:
                return ApiResponse(success=True, message="Spectrometer disconnected")
            else:
                return ApiResponse(success=False, message="Error disconnecting spectrometer")
        else:
            return ApiResponse(success=False, message="No hardware connected")

    except Exception as e:
        logger.error(f"Error disconnecting hardware: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/laser/on", response_model=ApiResponse)
async def laser_on():
    """Turn on the laser."""
    try:
        status = hardware_manager.get_spectrometer_status()

        if not status["connected"]:
            raise HTTPException(status_code=400, detail="Spectrometer not connected")

        # Note: B&W Tek hardware may not have direct laser control via serial
        # Laser is typically controlled by physical switch or acquisition commands
        success = hardware_manager.spectrometer.laser_on()

        if success:
            return ApiResponse(
                success=True,
                message="Laser control signal sent (check hardware for physical laser switch)",
            )
        else:
            return ApiResponse(success=False, message="Failed to control laser")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error turning on laser: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/laser/off", response_model=ApiResponse)
async def laser_off():
    """Turn off the laser."""
    try:
        status = hardware_manager.get_spectrometer_status()

        if not status["connected"]:
            raise HTTPException(status_code=400, detail="Spectrometer not connected")

        # Note: B&W Tek hardware may not have direct laser control via serial
        success = hardware_manager.spectrometer.laser_off()

        if success:
            return ApiResponse(
                success=True,
                message="Laser control signal sent (check hardware for physical laser switch)",
            )
        else:
            return ApiResponse(success=False, message="Failed to control laser")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error turning off laser: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ports", response_model=list)
async def scan_ports():
    """Scan for available serial ports."""
    try:
        ports = hardware_manager.scan_ports()
        logger.info(f"Found {len(ports)} available ports")
        return ports

    except Exception as e:
        logger.error(f"Error scanning ports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_demo_spectrum(integration_time: float) -> np.ndarray:
    """Generate a realistic demo spectrum."""
    # Create a synthetic Raman spectrum
    length = 2048
    x = np.arange(length)

    # Base spectrum
    spectrum = np.random.normal(100, 10, length)  # Noise baseline

    # Add some characteristic peaks (simulate different molecules based on time)
    peak_positions = [400, 800, 1200, 1600, 1800]
    peak_intensities = [500, 800, 600, 400, 300]

    for pos, intensity in zip(peak_positions, peak_intensities):
        if pos < length:
            width = 20
            peak = intensity * np.exp(-((x - pos) ** 2) / (2 * width**2))
            spectrum += peak

    # Add integration time effect (longer time = better SNR)
    snr_factor = np.sqrt(integration_time / 200.0)  # Normalize to 200ms
    spectrum = spectrum * snr_factor + np.random.normal(0, 50 / snr_factor, length)

    # Ensure non-negative
    spectrum = np.maximum(spectrum, 0)

    return spectrum
