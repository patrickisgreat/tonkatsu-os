"""
Hardware acquisition API routes for spectrometer control.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from tonkatsu_os.hardware import (
    HardwareManager,
    SpectrometerAcquisitionError,
    SpectrometerConnectionError,
    SpectrometerError,
)

from ..models import (
    AcquisitionRequest,
    AcquisitionResponse,
    ApiResponse,
    HardwareStatus,
)
from ..state import app_state

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_hardware_manager() -> HardwareManager:
    """Retrieve (or initialize) the shared hardware manager."""
    manager = app_state.get("hardware_manager")
    if manager is None:
        manager = HardwareManager()
        app_state["hardware_manager"] = manager
    return manager


def _average_spectra(spectra: list[np.ndarray]) -> np.ndarray:
    """Average multiple spectra with basic shape validation."""
    if not spectra:
        raise SpectrometerAcquisitionError("No spectra collected for averaging")
    length = len(spectra[0])
    for spectrum in spectra:
        if len(spectrum) != length:
            raise SpectrometerAcquisitionError("Spectra lengths differ during averaging")
    stacked = np.vstack(spectra)
    return np.mean(stacked, axis=0)


@router.post("/acquire", response_model=AcquisitionResponse)
async def acquire_spectrum(request: AcquisitionRequest):
    """
    Acquire a spectrum from the connected spectrometer.

    This endpoint returns real hardware data when available and
    simulator data when explicitly requested. Acquisition failures
    result in a 500 error with a descriptive reason.
    """
    manager = _get_hardware_manager()
    integration_time = int(request.integration_time)
    average_count = max(1, int(request.averages))

    try:
        spectra: list[np.ndarray] = []
        for _ in range(average_count):
            spectrum = manager.acquire_spectrum(
                integration_time,
                simulate=request.simulate,
                simulation_file=request.simulation_file,
            )
            spectra.append(np.asarray(spectrum, dtype=float))
        spectrum = _average_spectra(spectra) if average_count > 1 else spectra[0]
    except (SpectrometerConnectionError, SpectrometerAcquisitionError) as exc:
        logger.error("Spectrometer acquisition failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except SpectrometerError as exc:
        logger.error("Unexpected spectrometer error: %s", exc)
        raise HTTPException(status_code=500, detail="Spectrometer error") from exc

    status = manager.get_spectrometer_status()
    source = status.get("last_source") or ("simulator" if request.simulate else "hardware")
    acquired_at: datetime = status.get("last_acquired_at") or datetime.utcnow()

    return AcquisitionResponse(
        data=[float(x) for x in spectrum.tolist()],
        source=source,
        integration_time=float(integration_time),
        acquired_at=acquired_at,
        port=status.get("port"),
        simulation_file=status.get("simulation_file"),
        average_count=average_count,
    )


@router.get("/status", response_model=HardwareStatus)
async def get_hardware_status():
    """Get current hardware connection status."""
    manager = _get_hardware_manager()

    try:
        status = manager.get_spectrometer_status()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error getting hardware status: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    hardware_status = HardwareStatus(
        connected=bool(status.get("connected")),
        port=status.get("port"),
        laser_status="ready" if status.get("connected") else "disconnected",
        temperature=status.get("temperature"),
        last_communication=status.get("last_communication"),
        last_error=status.get("last_error"),
        last_source=status.get("last_source"),
        last_acquired_at=status.get("last_acquired_at"),
        simulate=bool(status.get("simulate")),
        simulation_file=status.get("simulation_file"),
        data_points=status.get("data_points"),
    )

    return hardware_status


@router.post("/connect", response_model=ApiResponse)
async def connect_hardware(
    port: Optional[str] = Query(
        None,
        description="Serial port for the spectrometer (ignored in simulator mode)",
    ),
    simulate: bool = Query(False, description="Use the simulator instead of hardware"),
    simulation_file: Optional[str] = Query(
        None, description="Optional path to recorded spectrum for simulation"
    ),
):
    """Connect to spectrometer hardware or initialize the simulator."""
    manager = _get_hardware_manager()
    target_port = port or ("simulator" if simulate else "/dev/ttyUSB0")

    try:
        manager.connect_spectrometer(
            port=target_port,
            simulate=simulate,
            simulation_file=simulation_file,
        )
    except SpectrometerConnectionError as exc:
        logger.error("Failed to connect to spectrometer: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except SpectrometerError as exc:  # pragma: no cover - defensive logging
        logger.error("Unexpected spectrometer error: %s", exc)
        raise HTTPException(status_code=500, detail="Spectrometer error") from exc

    message = (
        "Initialized spectrometer simulator"
        if simulate
        else f"Connected to B&W Tek spectrometer on {target_port}"
    )
    return ApiResponse(success=True, message=message)


@router.post("/disconnect", response_model=ApiResponse)
async def disconnect_hardware():
    """Disconnect from spectrometer hardware."""
    manager = _get_hardware_manager()
    status = manager.get_spectrometer_status()

    if not status.get("connected"):
        return ApiResponse(success=False, message="No spectrometer connected")

    try:
        success = manager.disconnect_spectrometer()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error disconnecting spectrometer: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if success:
        return ApiResponse(success=True, message="Spectrometer disconnected")
    return ApiResponse(success=False, message="Error disconnecting spectrometer")


@router.post("/laser/on", response_model=ApiResponse)
async def laser_on():
    """Turn on the laser, if the hardware supports it."""
    manager = _get_hardware_manager()
    status = manager.get_spectrometer_status()

    if not status.get("connected"):
        raise HTTPException(status_code=400, detail="Spectrometer not connected")

    spectrometer = manager.spectrometer
    if not spectrometer:
        raise HTTPException(status_code=400, detail="Spectrometer not initialized")

    try:
        success = spectrometer.laser_on()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error controlling laser: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if success:
        return ApiResponse(
            success=True,
            message="Laser control signal sent (verify hardware switch)",
        )
    return ApiResponse(success=False, message="Failed to control laser")


@router.post("/laser/off", response_model=ApiResponse)
async def laser_off():
    """Turn off the laser, if the hardware supports it."""
    manager = _get_hardware_manager()
    status = manager.get_spectrometer_status()

    if not status.get("connected"):
        raise HTTPException(status_code=400, detail="Spectrometer not connected")

    spectrometer = manager.spectrometer
    if not spectrometer:
        raise HTTPException(status_code=400, detail="Spectrometer not initialized")

    try:
        success = spectrometer.laser_off()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error controlling laser: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if success:
        return ApiResponse(
            success=True,
            message="Laser control signal sent (verify hardware switch)",
        )
    return ApiResponse(success=False, message="Failed to control laser")


@router.get("/ports", response_model=list)
async def scan_ports():
    """Scan for available serial ports."""
    manager = _get_hardware_manager()

    try:
        ports = manager.scan_ports()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error scanning ports: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ports


# ---------------------------------------------------------------------
# Dark subtraction endpoints
# ---------------------------------------------------------------------
@router.post("/dark/acquire", response_model=AcquisitionResponse)
async def acquire_dark(
    integration_time: Optional[int] = Query(
        None, description="Integration time in ms (uses default if not specified)"
    ),
    averages: int = Query(1, ge=1, le=50, description="Number of scans to average"),
):
    """
    Acquire a dark spectrum for background subtraction.

    Block the laser beam or remove the sample before calling this endpoint.
    The dark spectrum will be stored and can be subtracted from future acquisitions.
    """
    manager = _get_hardware_manager()
    spectrometer = manager.spectrometer

    if not spectrometer:
        raise HTTPException(status_code=400, detail="Spectrometer not connected")

    try:
        if averages > 1:
            integration_ms = int(
                integration_time or spectrometer.config.default_integration_time
            )
            spectra: list[np.ndarray] = []
            for _ in range(averages):
                spectrum = spectrometer.acquire_spectrum(integration_ms)
                spectra.append(np.asarray(spectrum, dtype=float))
            dark = _average_spectra(spectra)
            spectrometer.set_dark_spectrum(dark, integration_ms)
        else:
            dark = spectrometer.acquire_dark(integration_time)
    except (SpectrometerConnectionError, SpectrometerAcquisitionError) as exc:
        logger.error("Dark acquisition failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    status = manager.get_spectrometer_status()
    return AcquisitionResponse(
        data=[float(x) for x in dark.tolist()],
        source="dark",
        integration_time=float(integration_time or spectrometer.config.default_integration_time),
        acquired_at=datetime.utcnow(),
        port=status.get("port"),
        average_count=averages,
    )


@router.post("/dark/clear", response_model=ApiResponse)
async def clear_dark():
    """Clear the stored dark spectrum."""
    manager = _get_hardware_manager()
    spectrometer = manager.spectrometer

    if not spectrometer:
        raise HTTPException(status_code=400, detail="Spectrometer not connected")

    spectrometer.clear_dark()
    return ApiResponse(success=True, message="Dark spectrum cleared")


@router.get("/dark/info")
async def get_dark_info():
    """Get information about the stored dark spectrum."""
    manager = _get_hardware_manager()
    spectrometer = manager.spectrometer

    if not spectrometer:
        return {"has_dark": False, "error": "Spectrometer not connected"}

    return spectrometer.get_dark_info()


@router.post("/acquire/corrected", response_model=AcquisitionResponse)
async def acquire_spectrum_corrected(request: AcquisitionRequest):
    """
    Acquire a spectrum with automatic dark subtraction.

    If a dark spectrum has been acquired, it will be subtracted from the measurement.
    Otherwise, returns the raw spectrum.
    """
    manager = _get_hardware_manager()
    spectrometer = manager.spectrometer
    integration_time = int(request.integration_time)
    average_count = max(1, int(request.averages))

    if not spectrometer:
        raise HTTPException(status_code=400, detail="Spectrometer not connected")

    try:
        spectra: list[np.ndarray] = []
        for _ in range(average_count):
            spectrum = manager.acquire_spectrum(
                integration_time,
                simulate=request.simulate,
                simulation_file=request.simulation_file,
            )
            spectra.append(np.asarray(spectrum, dtype=float))
        spectrum = _average_spectra(spectra) if average_count > 1 else spectra[0]

        # Apply dark subtraction if available
        if spectrometer.has_dark():
            spectrum = spectrometer.apply_dark_subtraction(spectrum)
            dark_applied = True
        else:
            dark_applied = False

    except (SpectrometerConnectionError, SpectrometerAcquisitionError) as exc:
        logger.error("Spectrometer acquisition failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    status = manager.get_spectrometer_status()
    acquired_at: datetime = status.get("last_acquired_at") or datetime.utcnow()
    base_source = status.get("last_source") or ("simulator" if request.simulate else "hardware")
    source = "hardware_corrected" if dark_applied and base_source == "hardware" else base_source

    return AcquisitionResponse(
        data=[float(x) for x in spectrum.tolist()],
        source=source,
        integration_time=float(integration_time),
        acquired_at=acquired_at,
        port=status.get("port"),
        simulation_file=status.get("simulation_file"),
        average_count=average_count,
    )
