"""
Database API routes for spectrum management.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..models import (
    ApiResponse,
    DatabaseStats,
    SimilaritySearchRequest,
    SimilarSpectrum,
    SpectrumResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def get_database():
    """Dependency to get database instance."""
    from fastapi import Request

    from tonkatsu_os.database import RamanSpectralDatabase

    # In a real app, this would be injected properly
    return RamanSpectralDatabase()


def _extract_source_from_metadata(metadata):
    """Extract source from metadata, handling both dict and string formats."""
    if not metadata:
        return "unknown"
    
    # If metadata is a string representation of a dict, try to parse it
    if isinstance(metadata, str):
        try:
            import ast
            parsed_metadata = ast.literal_eval(metadata)
            return parsed_metadata.get("source", "unknown")
        except (ValueError, SyntaxError):
            return "unknown"
    
    # If metadata is already a dict
    if isinstance(metadata, dict):
        return metadata.get("source", "unknown")
    
    return "unknown"


def _parse_metadata(metadata):
    """Parse metadata from string to dict if needed."""
    if not metadata:
        return None
    
    # If metadata is a string representation of a dict, try to parse it
    if isinstance(metadata, str):
        try:
            import ast
            return ast.literal_eval(metadata)
        except (ValueError, SyntaxError):
            return {"raw": metadata}
    
    # If metadata is already a dict
    if isinstance(metadata, dict):
        return metadata
    
    return None


@router.get("/stats", response_model=DatabaseStats)
async def get_database_stats(db=Depends(get_database)):
    """Get database statistics."""
    try:
        stats = db.get_database_stats()
        return DatabaseStats(**stats)
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spectrum/{spectrum_id}", response_model=SpectrumResponse)
async def get_spectrum(spectrum_id: str, db=Depends(get_database)):
    """Get a specific spectrum by ID."""
    try:
        spectrum_data = db.get_spectrum_by_id(int(spectrum_id))
        if not spectrum_data:
            raise HTTPException(status_code=404, detail="Spectrum not found")

        return SpectrumResponse(
            id=str(spectrum_data["id"]),
            compound_name=spectrum_data["compound_name"],
            chemical_formula=spectrum_data.get("chemical_formula"),
            cas_number=spectrum_data.get("cas_number"),
            spectrum_data=spectrum_data["spectrum_data"].tolist(),
            preprocessed_spectrum=spectrum_data["preprocessed_spectrum"].tolist()
            if spectrum_data["preprocessed_spectrum"] is not None
            else None,
            peak_positions=spectrum_data["peak_positions"].tolist()
            if spectrum_data["peak_positions"] is not None
            else None,
            peak_intensities=spectrum_data["peak_intensities"].tolist()
            if spectrum_data["peak_intensities"] is not None
            else None,
            laser_wavelength=spectrum_data["laser_wavelength"],
            integration_time=spectrum_data["integration_time"],
            acquisition_date=spectrum_data["acquisition_date"],
            source=_extract_source_from_metadata(spectrum_data.get("metadata")),
            measurement_conditions=spectrum_data.get("measurement_conditions"),
            metadata=_parse_metadata(spectrum_data.get("metadata")),
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid spectrum ID")
    except Exception as e:
        logger.error(f"Error getting spectrum {spectrum_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=List[SpectrumResponse])
async def search_spectra(
    q: str = Query(..., description="Search query"),
    exact_match: bool = Query(False, description="Exact match for compound name"),
    db=Depends(get_database),
):
    """Search spectra by compound name."""
    try:
        # First get basic results
        basic_results = db.search_by_compound_name(q, exact_match=exact_match)
        
        # Then get full spectrum data for each result
        full_results = []
        for result in basic_results:
            spectrum_data = db.get_spectrum_by_id(result["id"])
            if spectrum_data:
                full_results.append(SpectrumResponse(
                    id=str(spectrum_data["id"]),
                    compound_name=spectrum_data["compound_name"],
                    chemical_formula=spectrum_data.get("chemical_formula"),
                    cas_number=spectrum_data.get("cas_number"),
                    spectrum_data=spectrum_data["spectrum_data"].tolist(),
                    preprocessed_spectrum=spectrum_data["preprocessed_spectrum"].tolist()
                    if spectrum_data["preprocessed_spectrum"] is not None
                    else None,
                    peak_positions=spectrum_data["peak_positions"].tolist()
                    if spectrum_data["peak_positions"] is not None
                    else None,
                    peak_intensities=spectrum_data["peak_intensities"].tolist()
                    if spectrum_data["peak_intensities"] is not None
                    else None,
                    laser_wavelength=spectrum_data["laser_wavelength"],
                    integration_time=spectrum_data["integration_time"],
                    acquisition_date=spectrum_data["acquisition_date"],
                    source=_extract_source_from_metadata(spectrum_data.get("metadata")),
                    measurement_conditions=spectrum_data.get("measurement_conditions"),
                    metadata=_parse_metadata(spectrum_data.get("metadata")),
                ))
        
        return full_results
    except Exception as e:
        logger.error(f"Error searching spectra: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similar", response_model=List[SimilarSpectrum])
async def find_similar_spectra(request: SimilaritySearchRequest, db=Depends(get_database)):
    """Find similar spectra using similarity search."""
    try:
        import numpy as np

        spectrum_array = np.array(request.spectrum_data)

        similar_spectra = db.search_similar_spectra(
            spectrum_array, top_k=request.top_k, similarity_threshold=request.similarity_threshold
        )

        return [
            SimilarSpectrum(
                spectrum_id=str(spec["spectrum_id"]),
                compound_name=spec["compound_name"],
                chemical_formula=spec.get("chemical_formula"),
                cas_number=spec.get("cas_number"),
                similarity_score=spec["similarity_score"],
            )
            for spec in similar_spectra
        ]
    except Exception as e:
        logger.error(f"Error finding similar spectra: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compounds", response_model=List[str])
async def get_compounds(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of compounds to return"),
    db=Depends(get_database),
):
    """Get list of unique compounds in database."""
    try:
        stats = db.get_database_stats()
        compounds = [compound[0] for compound in stats["top_compounds"][:limit]]
        return compounds
    except Exception as e:
        logger.error(f"Error getting compounds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/spectrum/{spectrum_id}", response_model=ApiResponse)
async def delete_spectrum(spectrum_id: str, db=Depends(get_database)):
    """Delete a spectrum from the database."""
    try:
        # This would need to be implemented in the database class
        # For now, return a placeholder response
        return ApiResponse(success=False, error="Delete functionality not implemented yet")
    except Exception as e:
        logger.error(f"Error deleting spectrum {spectrum_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compounds/{compound_name}/spectra", response_model=List[dict])
async def get_compound_spectra(
    compound_name: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of spectra to return"),
    db=Depends(get_database),
):
    """Get all spectra for a specific compound."""
    try:
        results = db.search_by_compound_name(compound_name, exact_match=True)
        return results[:limit]
    except Exception as e:
        logger.error(f"Error getting spectra for compound {compound_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
