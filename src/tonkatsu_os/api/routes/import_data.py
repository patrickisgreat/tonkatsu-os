"""
Data import and integration API routes.
"""

import json
import logging
import os
import tempfile
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from tonkatsu_os.core import DataIntegrator, SyntheticDataGenerator

from ..models import (
    DataIntegrationStatus,
    ImportOptions,
    ImportResult,
    NISTDownloadRequest,
    RRUFFDownloadRequest,
    SPECTRALDownloadRequest,
    SyntheticDataRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def get_database():
    """Dependency to get database instance."""
    from tonkatsu_os.database import RamanSpectralDatabase

    return RamanSpectralDatabase()


def get_data_integrator(db=Depends(get_database)):
    """Get data integrator instance."""
    return DataIntegrator(db)


@router.post("/spectrum", response_model=ImportResult)
async def import_single_spectrum(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    integrator: DataIntegrator = Depends(get_data_integrator),
):
    """Import a single spectrum file."""
    try:
        # Parse metadata
        metadata_dict = json.loads(metadata)

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        ) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Process the file
            from tonkatsu_os.core.spectrum_importer import SpectrumImporter
            from tonkatsu_os.database import RamanSpectralDatabase

            db = RamanSpectralDatabase()
            importer = SpectrumImporter(db)

            # Create mock file object
            class MockFile:
                def __init__(self, path, name):
                    self.name = name
                    self._path = path

                def read(self):
                    with open(self._path, "rb") as f:
                        return f.read()

            mock_file = MockFile(temp_file_path, file.filename)
            spectrum_data = importer._parse_uploaded_file(mock_file)

            if spectrum_data:
                # Add metadata
                spectrum_data.update(metadata_dict)

                # Import spectrum
                spectrum_id = importer._import_single_spectrum(spectrum_data)

                if spectrum_id:
                    return ImportResult(
                        total_processed=1, successful_integrations=1, errors=0, error_messages=[]
                    )
                else:
                    return ImportResult(
                        total_processed=1,
                        successful_integrations=0,
                        errors=1,
                        error_messages=["Failed to import spectrum"],
                    )
            else:
                return ImportResult(
                    total_processed=1,
                    successful_integrations=0,
                    errors=1,
                    error_messages=["Failed to parse spectrum file"],
                )

        finally:
            # Clean up temp file
            os.unlink(temp_file_path)

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")
    except Exception as e:
        logger.error(f"Error importing spectrum: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=ImportResult)
async def import_batch_spectra(
    files: List[UploadFile] = File(...),
    options: str = Form(...),
    integrator: DataIntegrator = Depends(get_data_integrator),
):
    """Import multiple spectrum files in batch."""
    try:
        # Parse options
        options_dict = json.loads(options)
        import_options = ImportOptions(**options_dict)

        # Process files
        spectra_list = []
        errors = []

        for file in files:
            try:
                # Save temporarily and process
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(file.filename)[1]
                ) as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                try:
                    from tonkatsu_os.core.spectrum_importer import SpectrumImporter
                    from tonkatsu_os.database import RamanSpectralDatabase

                    db = RamanSpectralDatabase()
                    importer = SpectrumImporter(db)

                    class MockFile:
                        def __init__(self, path, name):
                            self.name = name
                            self._path = path

                        def read(self):
                            with open(self._path, "rb") as f:
                                return f.read()

                    mock_file = MockFile(temp_file_path, file.filename)
                    spectrum_data = importer._parse_uploaded_file(mock_file)

                    if spectrum_data:
                        spectrum_data["auto_preprocess"] = import_options.auto_preprocess
                        spectrum_data["validate_data"] = import_options.validate_data
                        spectra_list.append(spectrum_data)
                    else:
                        errors.append(f"Failed to parse {file.filename}")

                finally:
                    os.unlink(temp_file_path)

            except Exception as e:
                errors.append(f"Error processing {file.filename}: {str(e)}")

        # Import all spectra
        if spectra_list:
            result = integrator.integrate_spectra(spectra_list)
            result["error_messages"].extend(errors)
            return ImportResult(**result)
        else:
            return ImportResult(
                total_processed=len(files),
                successful_integrations=0,
                errors=len(files),
                error_messages=errors,
            )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid options JSON")
    except Exception as e:
        logger.error(f"Error in batch import: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rruff/download", response_model=ImportResult)
async def download_rruff_data(
    request: RRUFFDownloadRequest, integrator: DataIntegrator = Depends(get_data_integrator)
):
    """
    Download and integrate RRUFF database spectra.

    This automatically downloads Raman spectra from the RRUFF mineral database
    and integrates them into your local database. No manual downloading required!
    """
    try:
        logger.info(f"Starting RRUFF download for {request.max_spectra} spectra")

        result = integrator.download_and_integrate_rruff(max_spectra=request.max_spectra)

        # Handle case where result might not have the expected structure
        if isinstance(result, dict) and 'successful_integrations' in result:
            logger.info(f"RRUFF download completed: {result['successful_integrations']} spectra added")
            return ImportResult(**result)
        else:
            # Fallback for compatibility
            logger.warning("RRUFF download returned unexpected format, using fallback")
            return ImportResult(
                total_processed=0,
                successful_integrations=0,
                errors=1,
                error_messages=["RRUFF service temporarily unavailable - feature coming soon"]
            )

    except Exception as e:
        logger.error(f"Error downloading RRUFF data: {e}")
        # Return a user-friendly error instead of throwing
        return ImportResult(
            total_processed=0,
            successful_integrations=0,
            errors=1,
            error_messages=[f"RRUFF download temporarily unavailable: {str(e)}"]
        )


@router.post("/nist/download", response_model=ImportResult)
async def download_nist_data(request: NISTDownloadRequest):
    """
    Download and integrate NIST database spectra.
    
    NIST Chemistry WebBook contains extensive spectral data including Raman spectra.
    """
    try:
        logger.info(f"Starting NIST download for {request.max_spectra} {request.spectral_type} spectra")
        
        # For now, return a placeholder response
        return ImportResult(
            total_processed=0,
            successful_integrations=0,
            errors=1,
            error_messages=["NIST database integration coming soon - feature in development"]
        )
        
    except Exception as e:
        logger.error(f"Error downloading NIST data: {e}")
        return ImportResult(
            total_processed=0,
            successful_integrations=0,
            errors=1,
            error_messages=[f"NIST download failed: {str(e)}"]
        )


@router.post("/spectral/download", response_model=ImportResult)
async def download_spectral_data(request: SPECTRALDownloadRequest):
    """
    Download and integrate public spectral database data.
    
    Access to various public spectral databases and repositories.
    """
    try:
        logger.info(f"Starting spectral database download for {request.max_spectra} spectra")
        
        # For now, return a placeholder response
        return ImportResult(
            total_processed=0,
            successful_integrations=0,
            errors=1,
            error_messages=["Public spectral database integration coming soon - feature in development"]
        )
        
    except Exception as e:
        logger.error(f"Error downloading spectral data: {e}")
        return ImportResult(
            total_processed=0,
            successful_integrations=0,
            errors=1,
            error_messages=[f"Spectral database download failed: {str(e)}"]
        )


@router.post("/rruff/chemistry/download", response_model=ImportResult)
async def download_rruff_chemistry_data(
    request: RRUFFDownloadRequest, integrator: DataIntegrator = Depends(get_data_integrator)
):
    """
    Download and integrate RRUFF chemistry microprobe data.
    
    This downloads chemistry analysis data from RRUFF microprobe measurements.
    """
    try:
        logger.info(f"Starting RRUFF chemistry download for {request.max_spectra} entries")
        
        from tonkatsu_os.core.data_loader import RRUFFDataLoader
        rruff_loader = RRUFFDataLoader()
        
        # Download chemistry data
        chemistry_data = rruff_loader.download_chemistry_data(max_spectra=request.max_spectra)
        
        if not chemistry_data:
            return ImportResult(
                total_processed=0,
                successful_integrations=0,
                errors=1,
                error_messages=["Failed to download any RRUFF chemistry data"]
            )
        
        # Integrate into database
        return integrator.integrate_spectra(chemistry_data)
        
    except Exception as e:
        logger.error(f"Error downloading RRUFF chemistry data: {e}")
        return ImportResult(
            total_processed=0,
            successful_integrations=0,
            errors=1,
            error_messages=[f"RRUFF chemistry download failed: {str(e)}"]
        )


@router.post("/rruff/infrared/download", response_model=ImportResult)
async def download_rruff_infrared_data(
    request: RRUFFDownloadRequest, integrator: DataIntegrator = Depends(get_data_integrator)
):
    """
    Download and integrate RRUFF infrared spectroscopy data.
    
    This downloads processed infrared spectra from the RRUFF database.
    """
    try:
        logger.info(f"Starting RRUFF infrared download for {request.max_spectra} spectra")
        
        from tonkatsu_os.core.data_loader import RRUFFDataLoader
        rruff_loader = RRUFFDataLoader()
        
        # Download infrared spectra
        ir_spectra = rruff_loader.download_infrared_data(data_type="Processed", max_spectra=request.max_spectra)
        
        if not ir_spectra:
            return ImportResult(
                total_processed=0,
                successful_integrations=0,
                errors=1,
                error_messages=["Failed to download any RRUFF infrared spectra"]
            )
        
        # Integrate into database
        return integrator.integrate_spectra(ir_spectra)
        
    except Exception as e:
        logger.error(f"Error downloading RRUFF infrared data: {e}")
        return ImportResult(
            total_processed=0,
            successful_integrations=0,
            errors=1,
            error_messages=[f"RRUFF infrared download failed: {str(e)}"]
        )


@router.post("/pharmaceutical/download", response_model=ImportResult)
async def download_pharmaceutical_data(
    request: RRUFFDownloadRequest, integrator: DataIntegrator = Depends(get_data_integrator)
):
    """
    Download and integrate pharmaceutical Raman spectroscopy data.
    
    This downloads open-source pharmaceutical ingredient Raman spectra data
    for active pharmaceutical ingredient (API) development and analysis.
    """
    try:
        logger.info(f"Starting pharmaceutical download for {request.max_spectra} spectra")
        
        from tonkatsu_os.core.data_loader import PharmaceuticalDataLoader
        pharma_loader = PharmaceuticalDataLoader()
        
        # Download pharmaceutical spectra
        pharma_spectra = pharma_loader.download_pharmaceutical_data(
            source="springer_nature", 
            max_spectra=request.max_spectra
        )
        
        if not pharma_spectra:
            return ImportResult(
                total_processed=0,
                successful_integrations=0,
                errors=1,
                error_messages=["Failed to download any pharmaceutical spectra"]
            )
        
        # Integrate into database
        return integrator.integrate_spectra(pharma_spectra)
        
    except Exception as e:
        logger.error(f"Error downloading pharmaceutical data: {e}")
        return ImportResult(
            total_processed=0,
            successful_integrations=0,
            errors=1,
            error_messages=[f"Pharmaceutical download failed: {str(e)}"]
        )


@router.post("/synthetic/generate", response_model=ImportResult)
async def generate_synthetic_data(
    request: SyntheticDataRequest, integrator: DataIntegrator = Depends(get_data_integrator)
):
    """
    Generate synthetic Raman spectra for testing and training.

    This creates artificial spectra for common compounds that can be used
    for testing the system and training machine learning models.
    """
    try:
        logger.info(
            f"Generating synthetic data: {request.samples_per_compound} samples per compound"
        )

        result = integrator.generate_and_integrate_synthetic(
            n_samples_per_compound=request.samples_per_compound
        )

        logger.info(
            f"Synthetic data generation completed: {result['successful_integrations']} spectra generated"
        )

        return ImportResult(**result)

    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate synthetic data: {str(e)}")


@router.get("/rruff/status", response_model=DataIntegrationStatus)
async def get_rruff_status(db=Depends(get_database)):
    """Get RRUFF database integration status."""
    try:
        # Check for RRUFF spectra in database
        stats = db.get_database_stats()

        # Count RRUFF spectra (those with 'RRUFF' in source metadata)
        rruff_count = 0
        synthetic_count = 0

        # This is a simplified check - in a real implementation,
        # you'd query the database for spectra with specific source metadata

        return DataIntegrationStatus(
            rruff_available=rruff_count > 0,
            rruff_last_update=None,
            rruff_count=rruff_count,
            synthetic_count=synthetic_count,
        )

    except Exception as e:
        logger.error(f"Error getting RRUFF status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/formats", response_model=List[str])
async def get_supported_formats():
    """Get list of supported import file formats."""
    return [".csv", ".txt", ".json", ".xlsx", ".tsv"]


@router.post("/validate", response_model=dict)
async def validate_spectrum_file(file: UploadFile = File(...)):
    """Validate a spectrum file without importing it."""
    try:
        # Save temporarily and validate
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        ) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            from tonkatsu_os.core.spectrum_importer import SpectrumImporter
            from tonkatsu_os.database import RamanSpectralDatabase

            db = RamanSpectralDatabase()
            importer = SpectrumImporter(db)

            class MockFile:
                def __init__(self, path, name):
                    self.name = name
                    self._path = path

                def read(self):
                    with open(self._path, "rb") as f:
                        return f.read()

            mock_file = MockFile(temp_file_path, file.filename)
            spectrum_data = importer._parse_uploaded_file(mock_file)

            if spectrum_data:
                return {
                    "valid": True,
                    "data_points": len(spectrum_data["spectrum_data"]),
                    "compound_name": spectrum_data.get("compound_name", "Unknown"),
                    "format_detected": os.path.splitext(file.filename)[1],
                    "message": "File is valid and ready for import",
                }
            else:
                return {"valid": False, "message": "Could not parse spectrum data from file"}

        finally:
            os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Error validating file: {e}")
        return {"valid": False, "message": f"Validation error: {str(e)}"}
