"""
Reference spectrum lookup endpoints.
"""

import logging
from typing import Any, Callable, Dict, List, Set

from fastapi import APIRouter, HTTPException, Request

from tonkatsu_os.database import RamanSpectralDatabase

from ..models import ReferenceSpectrum, ReferenceSpectrumRequest

router = APIRouter()
logger = logging.getLogger(__name__)


def _resolve_dependency(request: Request, accessor: str, fallback: Callable[[], Any]) -> Any:
    getter = getattr(request.app, accessor, None)
    if callable(getter):
        try:
            return getter()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to obtain '%s' from app state: %s", accessor, exc)
    return fallback()


def _resolve_database(request: Request) -> RamanSpectralDatabase:
    return _resolve_dependency(
        request,
        "get_database",
        lambda: RamanSpectralDatabase(),
    )


def _serialize_spectrum(row: Dict[str, Any], include_preprocessed: bool) -> ReferenceSpectrum:
    spectrum_data = row.get("spectrum_data")
    if spectrum_data is None:
        raise HTTPException(status_code=500, detail="Spectrum missing raw data buffer")

    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            import ast

            metadata = ast.literal_eval(metadata)
        except Exception:  # pragma: no cover - best effort
            metadata = {"raw": metadata}

    preprocessed = row.get("preprocessed_spectrum")
    if not include_preprocessed:
        preprocessed = None

    return ReferenceSpectrum(
        id=row.get("id"),
        compound_name=row.get("compound_name"),
        chemical_formula=row.get("chemical_formula"),
        cas_number=row.get("cas_number"),
        measurement_conditions=row.get("measurement_conditions"),
        laser_wavelength=row.get("laser_wavelength"),
        integration_time=row.get("integration_time"),
        acquisition_date=row.get("acquisition_date"),
        spectrum_data=spectrum_data.tolist() if hasattr(spectrum_data, "tolist") else list(spectrum_data),
        preprocessed_spectrum=preprocessed.tolist() if hasattr(preprocessed, "tolist") else preprocessed,
        metadata=metadata,
    )


@router.post("/spectra", response_model=List[ReferenceSpectrum])
async def fetch_reference_spectra(
    request_data: ReferenceSpectrumRequest,
    http_request: Request,
) -> List[ReferenceSpectrum]:
    """
    Retrieve reference spectra using compound hints or functional groups.

    Currently this searches the local Raman database for compound names containing the provided hints.
    Future revisions will enrich results with external reference libraries.
    """
    hints = [hint.strip() for hint in request_data.hints if hint.strip()]
    functional_groups = (
        [group.strip() for group in (request_data.functional_groups or []) if group.strip()]
        if request_data.functional_groups
        else []
    )

    if not hints and not functional_groups:
        raise HTTPException(status_code=400, detail="Provide at least one hint or functional group")

    database = _resolve_database(http_request)

    seen_ids: Set[int] = set()
    collected: List[ReferenceSpectrum] = []

    try:
        # Prioritize direct hints
        for hint in hints:
            for match in database.search_by_compound_name(hint):
                spectrum_id = int(match["id"])
                if spectrum_id in seen_ids:
                    continue

                row = database.get_spectrum_by_id(spectrum_id)
                if not row:
                    continue

                collected.append(_serialize_spectrum(row, request_data.include_preprocessed))
                seen_ids.add(spectrum_id)

                if len(collected) >= request_data.limit:
                    return collected

        # Functional groups can be stored under metadata['functional_groups']
        if functional_groups and len(collected) < request_data.limit:
            remaining = request_data.limit - len(collected)
            cursor = database.conn.execute(
                """
                SELECT id, compound_name, chemical_formula
                FROM spectra
                WHERE metadata IS NOT NULL
                """
            )
            for spectrum_id, compound_name, formula in cursor.fetchall():
                if spectrum_id in seen_ids:
                    continue

                row = database.get_spectrum_by_id(int(spectrum_id))
                if not row:
                    continue

                metadata = row.get("metadata")
                groups = []
                if isinstance(metadata, str):
                    try:
                        import ast

                        metadata = ast.literal_eval(metadata)
                    except Exception:
                        metadata = {"raw": metadata}
                if isinstance(metadata, dict):
                    groups = metadata.get("functional_groups") or metadata.get("functionalGroups") or []

                matched = False
                for group in functional_groups:
                    if any(group.lower() in str(item).lower() for item in groups):
                        matched = True
                        break

                if not matched:
                    continue

                collected.append(_serialize_spectrum(row, request_data.include_preprocessed))
                seen_ids.add(int(spectrum_id))

                if len(collected) >= request_data.limit:
                    break

        return collected[: request_data.limit]
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Reference spectrum lookup failed: %s", exc)
        raise HTTPException(status_code=500, detail="Reference lookup failed") from exc
