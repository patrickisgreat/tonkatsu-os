"""
Hardware acquisition API routes for spectrometer control.
"""

from fastapi import APIRouter, HTTPException
import logging
import numpy as np
import time

from ..models import AcquisitionRequest, HardwareStatus, ApiResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Mock hardware state
hardware_state = {
    "connected": False,
    "port": None,
    "laser_status": "off",
    "temperature": 25.0,
    "last_communication": None
}

@router.post("/acquire", response_model=list)
async def acquire_spectrum(request: AcquisitionRequest):
    """
    Acquire a spectrum from the connected spectrometer.
    
    If no hardware is connected, this generates a synthetic spectrum
    for demonstration purposes.
    """
    try:
        logger.info(f"Acquiring spectrum with integration time: {request.integration_time}ms")
        
        # Check if hardware is connected
        if not hardware_state["connected"]:
            # Generate synthetic spectrum for demo
            logger.info("No hardware connected, generating synthetic spectrum")
            spectrum = _generate_demo_spectrum(request.integration_time)
        else:
            # In production, this would interface with actual hardware
            spectrum = _acquire_from_hardware(request)
        
        return spectrum.tolist()
        
    except Exception as e:
        logger.error(f"Error acquiring spectrum: {e}")
        raise HTTPException(status_code=500, detail=f"Acquisition failed: {str(e)}")

@router.get("/status", response_model=HardwareStatus)
async def get_hardware_status():
    """Get current hardware connection status."""
    try:
        from datetime import datetime
        
        # Try to detect connected hardware
        hardware_connected = _check_hardware_connection()
        
        if hardware_connected:
            hardware_state.update({
                "connected": True,
                "port": "/dev/ttyUSB0",
                "laser_status": "ready",
                "temperature": 25.0 + np.random.normal(0, 1),  # Simulate temperature variation
                "last_communication": datetime.now()
            })
        else:
            hardware_state.update({
                "connected": False,
                "port": None,
                "laser_status": "disconnected",
                "temperature": None,
                "last_communication": None
            })
        
        return HardwareStatus(**hardware_state)
        
    except Exception as e:
        logger.error(f"Error getting hardware status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connect", response_model=ApiResponse)
async def connect_hardware(port: str = "/dev/ttyUSB0"):
    """Connect to spectrometer hardware."""
    try:
        logger.info(f"Attempting to connect to hardware on port: {port}")
        
        # In production, this would attempt actual hardware connection
        # For demo, we'll simulate connection
        success = _attempt_hardware_connection(port)
        
        if success:
            hardware_state.update({
                "connected": True,
                "port": port,
                "laser_status": "ready"
            })
            
            return ApiResponse(
                success=True,
                message=f"Connected to spectrometer on {port}"
            )
        else:
            return ApiResponse(
                success=False,
                message=f"Failed to connect to hardware on {port}"
            )
        
    except Exception as e:
        logger.error(f"Error connecting to hardware: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/disconnect", response_model=ApiResponse)
async def disconnect_hardware():
    """Disconnect from spectrometer hardware."""
    try:
        if hardware_state["connected"]:
            # In production, this would close hardware connections
            hardware_state.update({
                "connected": False,
                "port": None,
                "laser_status": "off",
                "temperature": None
            })
            
            return ApiResponse(
                success=True,
                message="Hardware disconnected"
            )
        else:
            return ApiResponse(
                success=False,
                message="No hardware connected"
            )
        
    except Exception as e:
        logger.error(f"Error disconnecting hardware: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/laser/on", response_model=ApiResponse)
async def laser_on():
    """Turn on the laser."""
    try:
        if not hardware_state["connected"]:
            raise HTTPException(status_code=400, detail="Hardware not connected")
        
        hardware_state["laser_status"] = "on"
        
        return ApiResponse(
            success=True,
            message="Laser turned on"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error turning on laser: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/laser/off", response_model=ApiResponse)
async def laser_off():
    """Turn off the laser."""
    try:
        if not hardware_state["connected"]:
            raise HTTPException(status_code=400, detail="Hardware not connected")
        
        hardware_state["laser_status"] = "off"
        
        return ApiResponse(
            success=True,
            message="Laser turned off"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error turning off laser: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _check_hardware_connection() -> bool:
    """Check if hardware is connected."""
    try:
        # In production, this would check serial ports, etc.
        # For demo, return False (no hardware connected)
        return False
    except:
        return False

def _attempt_hardware_connection(port: str) -> bool:
    """Attempt to connect to hardware on specified port."""
    try:
        # In production, this would try to open serial connection
        # For demo, simulate connection attempt
        import os
        return os.path.exists(port) if port.startswith('/dev/') else False
    except:
        return False

def _acquire_from_hardware(request: AcquisitionRequest) -> np.ndarray:
    """Acquire spectrum from actual hardware."""
    # This would interface with the actual spectrometer
    # For now, return synthetic data
    return _generate_demo_spectrum(request.integration_time)

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
            peak = intensity * np.exp(-((x - pos) ** 2) / (2 * width ** 2))
            spectrum += peak
    
    # Add integration time effect (longer time = better SNR)
    snr_factor = np.sqrt(integration_time / 200.0)  # Normalize to 200ms
    spectrum = spectrum * snr_factor + np.random.normal(0, 50/snr_factor, length)
    
    # Ensure non-negative
    spectrum = np.maximum(spectrum, 0)
    
    return spectrum