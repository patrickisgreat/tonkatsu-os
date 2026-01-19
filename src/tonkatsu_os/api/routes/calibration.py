"""
Calibration API routes for instrument-specific axis mapping.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from ..models import (
    ApiResponse,
    CalibrationCreate,
    CalibrationDetail,
    CalibrationSummary,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _resolve_database(request: Request):
    getter = getattr(request.app, "get_database", None)
    if callable(getter):
        return getter()
    from tonkatsu_os.database.raman_database import RamanSpectralDatabase

    return RamanSpectralDatabase()


@router.post("/create", response_model=CalibrationDetail)
async def create_calibration(payload: CalibrationCreate, request: Request):
    db = _resolve_database(request)
    try:
        calibration_id = db.add_calibration(
            name=payload.name,
            instrument_id=payload.instrument_id,
            axis_data=payload.axis_data,
            laser_wavelength=payload.laser_wavelength,
            notes=payload.notes,
            set_active=payload.set_active,
        )
        calibration = db.get_calibration(calibration_id)
        if not calibration:
            raise HTTPException(status_code=500, detail="Failed to load calibration")
        return CalibrationDetail(**calibration)
    except Exception as exc:
        logger.error("Failed to create calibration: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/list", response_model=list[CalibrationSummary])
async def list_calibrations(
    request: Request,
    instrument_id: Optional[str] = Query(default=None),
):
    db = _resolve_database(request)
    calibrations = db.list_calibrations(instrument_id=instrument_id)
    return [CalibrationSummary(**item) for item in calibrations]


@router.get("/active", response_model=CalibrationDetail)
async def get_active_calibration(
    request: Request,
    instrument_id: str = Query(...),
):
    db = _resolve_database(request)
    calibration = db.get_active_calibration(instrument_id=instrument_id)
    if not calibration:
        raise HTTPException(status_code=404, detail="No active calibration found")
    return CalibrationDetail(**calibration)


@router.get("/{calibration_id}", response_model=CalibrationDetail)
async def get_calibration(calibration_id: int, request: Request):
    db = _resolve_database(request)
    calibration = db.get_calibration(calibration_id)
    if not calibration:
        raise HTTPException(status_code=404, detail="Calibration not found")
    return CalibrationDetail(**calibration)


@router.post("/activate/{calibration_id}", response_model=ApiResponse)
async def activate_calibration(
    calibration_id: int,
    request: Request,
    instrument_id: str = Query(...),
):
    db = _resolve_database(request)
    success = db.set_active_calibration(calibration_id, instrument_id=instrument_id)
    if not success:
        raise HTTPException(status_code=404, detail="Calibration not found")
    return ApiResponse(success=True, message="Calibration activated")
