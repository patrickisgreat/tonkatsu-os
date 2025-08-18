"""
Pydantic models for API request/response schemas.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# Base response model
class ApiResponse(BaseModel):
    success: bool = True
    message: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None


# Spectrum models
class SpectrumData(BaseModel):
    spectrum_data: List[float] = Field(..., description="Raw spectrum intensity values")


class SpectrumMetadata(BaseModel):
    compound_name: str = Field(..., description="Name of the compound")
    chemical_formula: Optional[str] = Field(None, description="Chemical formula")
    cas_number: Optional[str] = Field(None, description="CAS registry number")
    laser_wavelength: float = Field(473.0, description="Laser wavelength in nm")
    integration_time: float = Field(200.0, description="Integration time in ms")
    measurement_conditions: Optional[str] = Field(None, description="Measurement conditions")
    source: str = Field("api_upload", description="Data source")


class SpectrumCreate(SpectrumData, SpectrumMetadata):
    """Model for creating a new spectrum."""

    pass


class SpectrumResponse(BaseModel):
    id: str
    compound_name: str
    chemical_formula: Optional[str]
    cas_number: Optional[str]
    spectrum_data: List[float]
    preprocessed_spectrum: Optional[List[float]]
    peak_positions: Optional[List[int]]
    peak_intensities: Optional[List[float]]
    laser_wavelength: Optional[float]
    integration_time: Optional[float]
    acquisition_date: datetime
    source: str
    measurement_conditions: Optional[str]
    metadata: Optional[Dict[str, Any]]


# Analysis models
class AnalysisRequest(BaseModel):
    spectrum_data: List[float] = Field(..., description="Spectrum data to analyze")
    preprocess: bool = Field(True, description="Whether to preprocess the spectrum")


class PredictionResult(BaseModel):
    compound: str
    probability: float


class ModelPrediction(BaseModel):
    compound: str
    confidence: float


class IndividualPredictions(BaseModel):
    random_forest: ModelPrediction
    svm: ModelPrediction
    neural_network: ModelPrediction
    pls_regression: Optional[ModelPrediction] = None


class ConfidenceComponents(BaseModel):
    probability_score: float
    entropy_score: float
    peak_match_score: float
    model_agreement_score: float
    spectral_quality_score: float


class ConfidenceAnalysis(BaseModel):
    overall_confidence: float
    confidence_components: ConfidenceComponents
    risk_level: str = Field(..., pattern="^(low|medium|high)$")
    recommendation: str


class AnalysisResult(BaseModel):
    predicted_compound: str
    confidence: float
    uncertainty: float
    model_agreement: float
    top_predictions: List[PredictionResult]
    individual_predictions: IndividualPredictions
    confidence_analysis: ConfidenceAnalysis
    processing_time: float
    
    # New fields for enhanced workflow
    method: Optional[str] = Field(None, description="Analysis method used (database_similarity, external_api_nist, etc.)")
    database_matches_found: Optional[int] = Field(None, description="Number of database matches found")
    similarity_threshold_used: Optional[float] = Field(None, description="Similarity threshold used for database search")
    fallback_reason: Optional[str] = Field(None, description="Reason for using fallback method")
    database_match_details: Optional[Dict[str, Any]] = Field(None, description="Details of database matches")
    external_api_details: Optional[Dict[str, Any]] = Field(None, description="Details of external API calls")


# Database models
class DatabaseStats(BaseModel):
    total_spectra: int
    unique_compounds: int
    top_compounds: List[List[Union[str, int]]]


class SimilarSpectrum(BaseModel):
    spectrum_id: str
    compound_name: str
    chemical_formula: Optional[str]
    cas_number: Optional[str]
    similarity_score: float


class SimilaritySearchRequest(BaseModel):
    spectrum_data: List[float]
    top_k: int = Field(5, ge=1, le=20, description="Number of similar spectra to return")
    similarity_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )


# Import models
class ImportOptions(BaseModel):
    auto_preprocess: bool = Field(True, description="Automatically preprocess spectra")
    validate_data: bool = Field(True, description="Validate data quality")
    overwrite_existing: bool = Field(False, description="Overwrite existing entries")
    generate_features: bool = Field(True, description="Generate ML features")


class ImportResult(BaseModel):
    total_processed: int
    successful_integrations: int
    errors: int
    error_messages: List[str]


# Training models
class TrainingConfig(BaseModel):
    use_pca: bool = Field(True, description="Use PCA for dimensionality reduction")
    n_components: int = Field(50, ge=10, le=200, description="Number of PCA components")
    validation_split: float = Field(0.2, ge=0.1, le=0.4, description="Validation split ratio")
    optimize_hyperparams: bool = Field(False, description="Optimize hyperparameters")


class TrainingResult(BaseModel):
    rf_accuracy: float
    svm_accuracy: float
    nn_accuracy: float
    pls_accuracy: Optional[float] = None
    ensemble_accuracy: float
    n_train_samples: int
    n_val_samples: int
    n_features: int
    n_classes: int
    classification_report: Dict[str, Any]
    confusion_matrix: List[List[int]]
    training_time: float


class TrainingStatus(BaseModel):
    is_training: bool
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    model_exists: bool
    last_trained: Optional[datetime]


# Acquisition models
class AcquisitionRequest(BaseModel):
    integration_time: float = Field(200.0, ge=50.0, le=5000.0, description="Integration time in ms")
    laser_power: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Laser power percentage"
    )


class HardwareStatus(BaseModel):
    connected: bool
    port: Optional[str]
    laser_status: Optional[str]
    temperature: Optional[float]
    last_communication: Optional[datetime]


# Data integration models
class RRUFFDownloadRequest(BaseModel):
    max_spectra: int = Field(50, ge=1, le=1000, description="Maximum number of spectra to download")


class NISTDownloadRequest(BaseModel):
    max_spectra: int = Field(50, ge=1, le=1000, description="Maximum number of spectra to download")
    spectral_type: str = Field("raman", description="Type of spectra (raman, ir, mass)")


class SPECTRALDownloadRequest(BaseModel):
    max_spectra: int = Field(50, ge=1, le=1000, description="Maximum number of spectra to download")
    compound_class: Optional[str] = Field(None, description="Filter by compound class")


class SyntheticDataRequest(BaseModel):
    samples_per_compound: int = Field(
        10, ge=1, le=100, description="Number of synthetic samples per compound"
    )
    compounds: Optional[List[str]] = Field(None, description="Specific compounds to generate")


class DataIntegrationStatus(BaseModel):
    rruff_available: bool
    rruff_last_update: Optional[datetime]
    rruff_count: int
    synthetic_count: int


# System models
class SystemHealth(BaseModel):
    status: str = Field(..., pattern="^(healthy|warning|error)$")
    components: Dict[str, bool]
    version: str
    uptime: Optional[float]
    memory_usage: Optional[float]


class ExportRequest(BaseModel):
    format: str = Field(..., pattern="^(csv|json|sqlite)$")
    include_metadata: bool = Field(True, description="Include metadata in export")
    compounds: Optional[List[str]] = Field(None, description="Specific compounds to export")


# Preprocessing models
class PreprocessingOptions(BaseModel):
    smooth: bool = Field(True, description="Apply smoothing")
    baseline_correct: bool = Field(True, description="Apply baseline correction")
    normalize: bool = Field(True, description="Normalize spectrum")
    remove_spikes: bool = Field(True, description="Remove cosmic ray spikes")
    smoothing_window: int = Field(11, ge=5, le=21, description="Smoothing window size")
    normalization_method: str = Field("minmax", pattern="^(minmax|standard|l2|area)$")


class PreprocessingRequest(BaseModel):
    spectrum_data: List[float]
    options: Optional[PreprocessingOptions] = None


# Peak detection models
class PeakDetectionOptions(BaseModel):
    height: Optional[float] = Field(None, description="Minimum peak height")
    prominence: float = Field(0.01, ge=0.001, le=1.0, description="Minimum peak prominence")
    distance: int = Field(10, ge=1, le=100, description="Minimum distance between peaks")
    width: Optional[int] = Field(None, description="Peak width constraints")


class PeakDetectionResult(BaseModel):
    peak_positions: List[int]
    peak_intensities: List[float]
    peak_properties: Dict[str, List[float]]
