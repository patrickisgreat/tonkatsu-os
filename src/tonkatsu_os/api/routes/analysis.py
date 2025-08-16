"""
Analysis API routes for spectrum processing and ML prediction.
"""

from fastapi import APIRouter, HTTPException, Depends
import logging
import time
import numpy as np

from ..models import (
    AnalysisRequest, AnalysisResult, PreprocessingRequest, 
    PeakDetectionResult, ApiResponse
)

router = APIRouter()
logger = logging.getLogger(__name__)

def get_preprocessor():
    """Dependency to get preprocessor instance."""
    from tonkatsu_os.preprocessing import AdvancedPreprocessor
    return AdvancedPreprocessor()

def get_classifier():
    """Dependency to get classifier instance."""
    from tonkatsu_os.ml import EnsembleClassifier
    return EnsembleClassifier()

@router.post("/analyze", response_model=AnalysisResult)
async def analyze_spectrum(
    request: AnalysisRequest,
    preprocessor=Depends(get_preprocessor),
    classifier=Depends(get_classifier)
):
    """
    Analyze a spectrum and predict the molecular compound.
    
    This is the main analysis endpoint that combines preprocessing,
    peak detection, and ML prediction to identify molecules.
    """
    try:
        start_time = time.time()
        
        spectrum_array = np.array(request.spectrum_data)
        
        # Preprocess if requested
        if request.preprocess:
            processed_spectrum = preprocessor.preprocess(spectrum_array)
        else:
            processed_spectrum = spectrum_array
        
        # Detect peaks
        peaks, peak_intensities = preprocessor.detect_peaks(processed_spectrum)
        
        # Extract features
        features = preprocessor.spectral_features(processed_spectrum)
        
        # For demo purposes, create mock analysis results
        # In production, this would use the actual trained classifier
        mock_result = _create_mock_analysis_result(processed_spectrum, peaks, features)
        
        processing_time = time.time() - start_time
        mock_result['processing_time'] = processing_time
        
        return AnalysisResult(**mock_result)
        
    except Exception as e:
        logger.error(f"Error analyzing spectrum: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/preprocess", response_model=list)
async def preprocess_spectrum(
    request: PreprocessingRequest,
    preprocessor=Depends(get_preprocessor)
):
    """Preprocess a spectrum with specified options."""
    try:
        spectrum_array = np.array(request.spectrum_data)
        
        if request.options:
            # Apply custom preprocessing options
            processed = spectrum_array.copy()
            
            if request.options.smooth:
                processed = preprocessor.smooth_spectrum(processed, 
                    window_length=request.options.smoothing_window)
            
            if request.options.baseline_correct:
                processed = preprocessor.baseline_correction(processed)
            
            if request.options.remove_spikes:
                processed = preprocessor.remove_cosmic_rays(processed)
            
            if request.options.normalize:
                processed = preprocessor.normalize_spectrum(processed, 
                    method=request.options.normalization_method)
        else:
            # Use default preprocessing
            processed = preprocessor.preprocess(spectrum_array)
        
        return processed.tolist()
        
    except Exception as e:
        logger.error(f"Error preprocessing spectrum: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@router.post("/peaks", response_model=PeakDetectionResult)
async def detect_peaks(
    spectrum_data: list,
    preprocessor=Depends(get_preprocessor)
):
    """Detect peaks in a spectrum."""
    try:
        spectrum_array = np.array(spectrum_data)
        peaks, peak_intensities = preprocessor.detect_peaks(spectrum_array)
        
        # Calculate peak properties
        peak_properties = preprocessor.calculate_peak_properties(spectrum_array, peaks)
        
        return PeakDetectionResult(
            peak_positions=peaks.tolist(),
            peak_intensities=peak_intensities.tolist(),
            peak_properties={
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in peak_properties.items()
            }
        )
        
    except Exception as e:
        logger.error(f"Error detecting peaks: {e}")
        raise HTTPException(status_code=500, detail=f"Peak detection failed: {str(e)}")

@router.post("/features", response_model=dict)
async def extract_features(
    spectrum_data: list,
    preprocessor=Depends(get_preprocessor)
):
    """Extract spectral features for ML analysis."""
    try:
        spectrum_array = np.array(spectrum_data)
        features = preprocessor.spectral_features(spectrum_array)
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

def _create_mock_analysis_result(spectrum: np.ndarray, peaks: np.ndarray, features: dict) -> dict:
    """Create mock analysis results for demonstration."""
    # This simulates what a real ML model would return
    
    # Mock predictions based on spectral characteristics
    if features['num_peaks'] > 10:
        primary_compound = "Benzene"
        confidence = 0.87
    elif features['spectral_centroid'] > 1000:
        primary_compound = "Ethanol" 
        confidence = 0.73
    elif len(peaks) < 5:
        primary_compound = "Water"
        confidence = 0.91
    else:
        primary_compound = "Unknown Organic"
        confidence = 0.45
    
    return {
        "predicted_compound": primary_compound,
        "confidence": confidence,
        "uncertainty": 1.0 - confidence,
        "model_agreement": 0.85,
        "top_predictions": [
            {"compound": primary_compound, "probability": confidence},
            {"compound": "Alternative 1", "probability": 0.12},
            {"compound": "Alternative 2", "probability": 0.08}
        ],
        "individual_predictions": {
            "random_forest": {"compound": primary_compound, "confidence": confidence + 0.02},
            "svm": {"compound": primary_compound, "confidence": confidence - 0.02},
            "neural_network": {"compound": primary_compound, "confidence": confidence}
        },
        "confidence_analysis": {
            "overall_confidence": confidence,
            "confidence_components": {
                "probability_score": confidence,
                "entropy_score": 0.92,
                "peak_match_score": 0.78,
                "model_agreement_score": 0.85,
                "spectral_quality_score": 0.95
            },
            "risk_level": "low" if confidence > 0.8 else "medium" if confidence > 0.6 else "high",
            "recommendation": "High confidence identification" if confidence > 0.8 else "Moderate confidence - consider validation"
        }
    }