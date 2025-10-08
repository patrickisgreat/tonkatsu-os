"""
Hardware acquisition API routes for spectrometer control.
"""

import logging
from datetime import datetime
from typing import Optional

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

    try:
        spectrum = manager.acquire_spectrum(
            integration_time,
            simulate=request.simulate,
            simulation_file=request.simulation_file,
        )
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
