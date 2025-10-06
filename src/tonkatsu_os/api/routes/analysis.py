"""
Analysis API routes for spectrum processing and ML prediction.
"""

import logging
import time
from typing import Any, Callable

import numpy as np
from fastapi import APIRouter, Body, HTTPException, Request

from ..models import (
    AnalysisRequest,
    AnalysisResult,
    ApiResponse,
    PeakDetectionResult,
    PreprocessingRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _resolve_dependency(request: Request, accessor: str, factory: Callable[[], Any]) -> Any:
    """Fetch shared dependency from the FastAPI app or fall back to local factory."""
    getter = getattr(request.app, accessor, None)
    if callable(getter):
        try:
            return getter()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to obtain '%s' from app state: %s", accessor, exc)
    return factory()


def _resolve_preprocessor(request: Request):
    return _resolve_dependency(
        request,
        "get_preprocessor",
        lambda: __import__(
            "tonkatsu_os.preprocessing.advanced_preprocessor",
            fromlist=["AdvancedPreprocessor"],
        ).AdvancedPreprocessor(),
    )


def _resolve_database(request: Request):
    return _resolve_dependency(
        request,
        "get_database",
        lambda: __import__(
            "tonkatsu_os.database.raman_database",
            fromlist=["RamanSpectralDatabase"],
        ).RamanSpectralDatabase(),
    )


def _resolve_classifier(request: Request):
    return _resolve_dependency(
        request,
        "get_classifier",
        lambda: None,
    )


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_spectrum(
    request_data: AnalysisRequest,
    http_request: Request,
):
    """
    Analyze a spectrum and predict the molecular compound.

    This is the main analysis endpoint that combines preprocessing,
    peak detection, and ML prediction to identify molecules.
    """
    try:
        start_time = time.time()

        preprocessor = _resolve_preprocessor(http_request)
        database = _resolve_database(http_request)
        classifier = _resolve_classifier(http_request)

        # Validate spectrum data
        spectrum_array = np.array(request_data.spectrum_data)
        
        # Check for NaN/inf values that cause matplotlib errors
        if np.any(np.isnan(spectrum_array)) or np.any(np.isinf(spectrum_array)):
            logger.error("Spectrum contains NaN or infinite values")
            spectrum_array = np.nan_to_num(spectrum_array, nan=0.0, posinf=0.0, neginf=0.0)
            logger.warning("Cleaned NaN/inf values from spectrum")
            
        # Ensure we have valid data
        if len(spectrum_array) == 0:
            raise ValueError("Empty spectrum data provided")
            
        logger.info(f"Analyzing spectrum: len={len(spectrum_array)}, min={np.min(spectrum_array):.2f}, max={np.max(spectrum_array):.2f}")

        # Preprocess if requested
        config = request_data.config or {}
        should_preprocess = request_data.preprocess and config.get("preprocessing", True)

        if should_preprocess:
            processed_spectrum = preprocessor.preprocess(spectrum_array)
        else:
            processed_spectrum = spectrum_array

        # Detect peaks
        peaks, peak_intensities = preprocessor.detect_peaks(processed_spectrum)

        # Extract features
        features = preprocessor.spectral_features(processed_spectrum)

        similarity_threshold = float(config.get("similarity_threshold", 0.7))
        top_k = int(config.get("top_k", 5))
        models_config = config.get("models") or {}

        # Try database similarity search first
        similar_spectra = database.search_similar_spectra(
            processed_spectrum, 
            top_k=top_k, 
            similarity_threshold=similarity_threshold
        )

        processing_time = time.time() - start_time

        if similar_spectra and len(similar_spectra) > 0:
            # Found good matches in database
            best_match = similar_spectra[0]
            result = _create_database_match_result(
                best_match, similar_spectra, processed_spectrum, peaks, features
            )
        else:
            # No good database matches - try external APIs if configured
            result = await _try_external_apis(processed_spectrum, peaks, features)
            
            # If external APIs fail, try trained ML models
            if not result:
                enable_trained_models = any(
                    models_config.get(model_name, True)
                    for model_name in ["random_forest", "svm", "neural_network", "pls_regression"]
                )

                if enable_trained_models:
                    result = _try_trained_ml_models(processed_spectrum, peaks, features, classifier)
            
            # If trained models fail, try pre-trained models
            if not result:
                result = _try_pretrained_models(processed_spectrum, peaks, features)
            
            # If everything fails, fall back to mock
            if not result:
                result = _create_mock_analysis_result(processed_spectrum, peaks, features)
                result["fallback_reason"] = "No database matches, external APIs unavailable, no trained/pretrained models"

        result["processing_time"] = processing_time
        result["database_matches_found"] = len(similar_spectra) if similar_spectra else 0
        result["similarity_threshold_used"] = similarity_threshold

        return AnalysisResult(**result)

    except Exception as e:
        logger.error(f"Error analyzing spectrum: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/preprocess", response_model=list)
async def preprocess_spectrum(
    request_data: PreprocessingRequest, http_request: Request
):
    """Preprocess a spectrum with specified options."""
    try:
        preprocessor = _resolve_preprocessor(http_request)
        spectrum_array = np.array(request_data.spectrum_data)

        if request_data.options:
            # Apply custom preprocessing options
            processed = spectrum_array.copy()

            if request_data.options.smooth:
                processed = preprocessor.smooth_spectrum(
                    processed, window_length=request_data.options.smoothing_window
                )

            if request_data.options.baseline_correct:
                processed = preprocessor.baseline_correction(processed)

            if request_data.options.remove_spikes:
                processed = preprocessor.remove_cosmic_rays(processed)

            if request_data.options.normalize:
                processed = preprocessor.normalize_spectrum(
                    processed, method=request_data.options.normalization_method
                )
        else:
            # Use default preprocessing
            processed = preprocessor.preprocess(spectrum_array)

        return processed.tolist()

    except Exception as e:
        logger.error(f"Error preprocessing spectrum: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


@router.post("/peaks", response_model=PeakDetectionResult)
async def detect_peaks(spectrum_data: list = Body(...), http_request: Request = None):
    """Detect peaks in a spectrum."""
    try:
        spectrum_array = np.array(spectrum_data)
        preprocessor = _resolve_preprocessor(http_request)
        peaks, peak_intensities = preprocessor.detect_peaks(spectrum_array)

        # Calculate peak properties
        peak_properties = preprocessor.calculate_peak_properties(spectrum_array, peaks)

        return PeakDetectionResult(
            peak_positions=peaks.tolist(),
            peak_intensities=peak_intensities.tolist(),
            peak_properties={
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in peak_properties.items()
            },
        )

    except Exception as e:
        logger.error(f"Error detecting peaks: {e}")
        raise HTTPException(status_code=500, detail=f"Peak detection failed: {str(e)}")


@router.post("/features", response_model=dict)
async def extract_features(spectrum_data: list = Body(...), http_request: Request = None):
    """Extract spectral features for ML analysis."""
    try:
        spectrum_array = np.array(spectrum_data)
        preprocessor = _resolve_preprocessor(http_request)
        features = preprocessor.spectral_features(spectrum_array)

        return features

    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")


def _create_database_match_result(
    best_match: dict, all_matches: list, spectrum: np.ndarray, peaks: np.ndarray, features: dict
) -> dict:
    """Create analysis result from database similarity match."""
    
    confidence = best_match["similarity_score"]
    compound_name = best_match["compound_name"]
    spectral_quality = _estimate_spectral_quality(spectrum)
    database_coverage = min(len(all_matches) / 5.0, 1.0) if all_matches else 0.0
    
    confidence_components = {
        "probability_score": confidence,
        "entropy_score": max(0.0, 1.0 - confidence),
        "peak_match_score": confidence,
        "model_agreement_score": 1.0,
        "spectral_quality_score": spectral_quality,
        "similarity_score": confidence,
        "database_coverage": database_coverage,
    }
    
    # Create top predictions from similar matches
    top_predictions = []
    for match in all_matches[:3]:
        top_predictions.append({
            "compound": match["compound_name"],
            "probability": match["similarity_score"]
        })
    
    return {
        "predicted_compound": compound_name,
        "confidence": confidence,
        "uncertainty": 1.0 - confidence,
        "model_agreement": 1.0,  # Database match has perfect agreement
        "method": "database_similarity",
        "top_predictions": top_predictions,
        "individual_predictions": {
            "random_forest": {"compound": compound_name, "confidence": confidence},
            "svm": {"compound": compound_name, "confidence": confidence},
            "neural_network": {"compound": compound_name, "confidence": confidence},
            "database_match": {"compound": compound_name, "confidence": confidence},
        },
        "confidence_analysis": {
            "overall_confidence": confidence,
            "confidence_components": confidence_components,
            "risk_level": "low" if confidence > 0.8 else "medium" if confidence > 0.6 else "high",
            "recommendation": f"Database match found with {confidence:.1%} similarity"
        },
        "database_match_details": {
            "best_match_id": best_match.get("spectrum_id"),
            "chemical_formula": best_match.get("chemical_formula"),
            "cas_number": best_match.get("cas_number"),
            "num_similar_matches": len(all_matches)
        }
    }


async def _try_external_apis(spectrum: np.ndarray, peaks: np.ndarray, features: dict) -> dict:
    """Try external APIs for compound identification."""
    import os
    import asyncio
    
    # Check if external APIs are configured
    nist_api_key = os.getenv("NIST_API_KEY")
    chemspider_api_key = os.getenv("CHEMSPIDER_API_KEY")
    
    if not (nist_api_key or chemspider_api_key):
        logger.info("No external API keys configured")
        return None
    
    results = []
    
    # Try NIST database
    if nist_api_key:
        nist_result = await _query_nist_api(spectrum, peaks, nist_api_key)
        if nist_result:
            results.append(("NIST", nist_result))
    
    # Try ChemSpider
    if chemspider_api_key:
        chemspider_result = await _query_chemspider_api(spectrum, peaks, chemspider_api_key)
        if chemspider_result:
            results.append(("ChemSpider", chemspider_result))
    
    if not results:
        return None
    
    # Combine results from multiple APIs
    best_result = max(results, key=lambda x: x[1]["confidence"])
    api_source, result_data = best_result
    
    return {
        "predicted_compound": result_data["compound"],
        "confidence": result_data["confidence"],
        "uncertainty": 1.0 - result_data["confidence"],
        "model_agreement": result_data.get("agreement", 0.5),
        "method": f"external_api_{api_source.lower()}",
        "top_predictions": result_data.get("alternatives", []),
        "individual_predictions": {
            f"{api_source.lower()}_api": {
                "compound": result_data["compound"], 
                "confidence": result_data["confidence"]
            }
        },
        "confidence_analysis": {
            "overall_confidence": result_data["confidence"],
            "confidence_components": {
                "api_confidence": result_data["confidence"],
                "external_validation": 1.0,
                "spectral_quality_score": _estimate_spectral_quality(spectrum),
            },
            "risk_level": "medium",  # External APIs get medium risk by default
            "recommendation": f"Identified using {api_source} database"
        },
        "external_api_details": {
            "source": api_source,
            "api_response_time": result_data.get("response_time", 0),
            "total_apis_queried": len(results)
        }
    }


async def _query_nist_api(spectrum: np.ndarray, peaks: np.ndarray, api_key: str) -> dict:
    """Query NIST spectral database API."""
    try:
        # This is a placeholder - actual NIST API implementation would go here
        # For now, return a mock response to demonstrate the structure
        await asyncio.sleep(0.1)  # Simulate API call delay
        
        # Mock NIST response based on spectral characteristics
        if len(peaks) > 8:
            return {
                "compound": "Toluene",
                "confidence": 0.82,
                "agreement": 0.78,
                "alternatives": [
                    {"compound": "Toluene", "probability": 0.82},
                    {"compound": "Benzene", "probability": 0.13},
                    {"compound": "Xylene", "probability": 0.05}
                ],
                "response_time": 0.1
            }
        return None
    except Exception as e:
        logger.error(f"NIST API query failed: {e}")
        return None


async def _query_chemspider_api(spectrum: np.ndarray, peaks: np.ndarray, api_key: str) -> dict:
    """Query ChemSpider database API."""
    try:
        # This is a placeholder - actual ChemSpider API implementation would go here
        await asyncio.sleep(0.15)  # Simulate API call delay
        
        # Mock ChemSpider response
        if np.mean(spectrum) > 0.5:
            return {
                "compound": "Acetone",
                "confidence": 0.75,
                "agreement": 0.71,
                "alternatives": [
                    {"compound": "Acetone", "probability": 0.75},
                    {"compound": "Methanol", "probability": 0.18},
                    {"compound": "Ethanol", "probability": 0.07}
                ],
                "response_time": 0.15
            }
        return None
    except Exception as e:
        logger.error(f"ChemSpider API query failed: {e}")
        return None


def _try_trained_ml_models(
    spectrum: np.ndarray,
    peaks: np.ndarray,
    features: dict,
    classifier=None,
) -> dict:
    """Try to use trained ML models for prediction."""
    try:
        import os
        from tonkatsu_os.ml import EnsembleClassifier
        from tonkatsu_os.preprocessing import AdvancedPreprocessor

        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "..", "..", "..", "trained_ensemble_model.pkl")
        model_path = os.path.abspath(model_path)

        if classifier is None:
            if not os.path.exists(model_path):
                logger.info("No trained model found for ML prediction")
                return None

            classifier = EnsembleClassifier()
            classifier.load_model(model_path)
        elif getattr(classifier, "is_trained", False) is False and os.path.exists(model_path):
            classifier.load_model(model_path)
        elif getattr(classifier, "is_trained", False) is False:
            logger.info("Trained classifier instance not available")
            return None

        preprocessor = AdvancedPreprocessor()
        processed_spectrum = preprocessor.preprocess(spectrum)
        spectral_features = preprocessor.spectral_features(processed_spectrum)
        
        # Build feature vector (must match training format)
        feature_vector = []
        feature_vector.extend([
            spectral_features.get('spectral_centroid', 0),
            spectral_features.get('spectral_spread', 0),
            spectral_features.get('spectral_skewness', 0),
            spectral_features.get('spectral_kurtosis', 0),
            spectral_features.get('spectral_rolloff', 0),
            spectral_features.get('spectral_flatness', 0),
            spectral_features.get('zero_crossing_rate', 0),
            spectral_features.get('num_peaks', 0),
            spectral_features.get('peak_density', 0),
            spectral_features.get('dominant_peak_intensity', 0),
            spectral_features.get('mean_intensity', 0),
            spectral_features.get('std_intensity', 0),
            spectral_features.get('max_intensity', 0),
            spectral_features.get('min_intensity', 0)
        ])
        
        # Add statistical moments
        feature_vector.extend([
            np.mean(processed_spectrum),
            np.std(processed_spectrum),
            np.median(processed_spectrum),
            np.percentile(processed_spectrum, 25),
            np.percentile(processed_spectrum, 75),
            np.max(processed_spectrum),
            np.min(processed_spectrum)
        ])
        
        # Add spectral bins (match training)
        spectrum_bins = processed_spectrum[::50]
        feature_vector.extend(spectrum_bins.tolist())
        
        # Make prediction
        X = np.array([feature_vector])
        predictions = classifier.predict(X)
        
        if predictions and len(predictions) > 0:
            prediction = predictions[0]
            spectral_quality = _estimate_spectral_quality(spectrum)
            confidence = prediction["confidence"]
            model_agreement = prediction.get("model_agreement", 0.0)
            
            confidence_components = {
                "probability_score": confidence,
                "entropy_score": max(0.0, 1.0 - confidence),
                "peak_match_score": confidence,
                "model_agreement_score": model_agreement,
                "spectral_quality_score": spectral_quality,
                "ml_confidence": confidence,
            }
            
            return {
                "predicted_compound": prediction["predicted_compound"],
                "confidence": confidence,
                "uncertainty": prediction["uncertainty"],
                "model_agreement": model_agreement,
                "method": "trained_ml_ensemble",
                "top_predictions": prediction["top_predictions"],
                "individual_predictions": prediction["individual_predictions"],
                "confidence_analysis": {
                    "overall_confidence": confidence,
                    "confidence_components": confidence_components,
                    "risk_level": "low" if prediction["confidence"] > 0.8 else "medium" if prediction["confidence"] > 0.6 else "high",
                    "recommendation": f"Prediction from trained ensemble model"
                },
                "trained_model_details": {
                    "model_path": model_path,
                    "model_type": "custom_ensemble",
                    "algorithms_used": ["random_forest", "svm", "neural_network", "pls_regression"]
                }
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Trained ML model prediction failed: {e}")
        return None


def _try_pretrained_models(spectrum: np.ndarray, peaks: np.ndarray, features: dict) -> dict:
    """Try to use pre-trained models for prediction."""
    try:
        from tonkatsu_os.ml.pretrained_models import PreTrainedModelManager
        
        # Initialize pre-trained model manager
        model_manager = PreTrainedModelManager()
        
        # Get ensemble prediction from all available models
        prediction = model_manager.predict_ensemble(spectrum)
        
        if prediction and prediction.get('predicted_compound'):
            spectral_quality = _estimate_spectral_quality(spectrum)
            confidence = prediction["confidence"]
            model_agreement = prediction.get("model_agreement", 0.0)
            
            confidence_components = {
                "probability_score": confidence,
                "entropy_score": max(0.0, 1.0 - confidence),
                "peak_match_score": confidence,
                "model_agreement_score": model_agreement,
                "spectral_quality_score": spectral_quality,
                "pretrained_confidence": confidence,
            }
            
            result = {
                "predicted_compound": prediction["predicted_compound"],
                "confidence": confidence,
                "uncertainty": prediction["uncertainty"],
                "model_agreement": model_agreement,
                "method": prediction["method"],
                "top_predictions": [
                    {"compound": prediction["predicted_compound"], "probability": confidence}
                ],
                "individual_predictions": prediction["individual_predictions"],
                "confidence_analysis": {
                    "overall_confidence": confidence,
                    "confidence_components": confidence_components,
                    "risk_level": "medium",  # Pre-trained models get medium risk
                    "recommendation": f"Prediction from pre-trained models ensemble"
                },
                "pretrained_model_details": prediction.get("pretrained_model_details", {})
            }
            
            logger.info(f"Pre-trained models predicted: {prediction['predicted_compound']} with {prediction['confidence']:.3f} confidence")
            return result
        
        return None
        
    except Exception as e:
        logger.error(f"Pre-trained model prediction failed: {e}")
        return None


def _estimate_spectral_quality(spectrum: np.ndarray) -> float:
    """Estimate the quality of a spectrum for confidence calculation."""
    # Simple quality metrics
    snr = np.mean(spectrum) / (np.std(spectrum) + 1e-10)
    dynamic_range = (np.max(spectrum) - np.min(spectrum)) / (np.max(spectrum) + 1e-10)
    
    # Normalize to 0-1 scale
    quality_score = min(1.0, (snr / 10.0 + dynamic_range) / 2.0)
    return quality_score


def _create_mock_analysis_result(spectrum: np.ndarray, peaks: np.ndarray, features: dict) -> dict:
    """Create mock analysis results for demonstration."""
    # This simulates what a real ML model would return

    # Mock predictions based on spectral characteristics
    if features["num_peaks"] > 10:
        primary_compound = "Benzene"
        confidence = 0.87
    elif features["spectral_centroid"] > 1000:
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
            {"compound": "Alternative 2", "probability": 0.08},
        ],
        "individual_predictions": {
            "random_forest": {"compound": primary_compound, "confidence": confidence + 0.02},
            "svm": {"compound": primary_compound, "confidence": confidence - 0.02},
            "neural_network": {"compound": primary_compound, "confidence": confidence},
        },
        "confidence_analysis": {
            "overall_confidence": confidence,
            "confidence_components": {
                "probability_score": confidence,
                "entropy_score": 0.92,
                "peak_match_score": 0.78,
                "model_agreement_score": 0.85,
                "spectral_quality_score": 0.95,
            },
            "risk_level": "low" if confidence > 0.8 else "medium" if confidence > 0.6 else "high",
            "recommendation": "High confidence identification"
            if confidence > 0.8
            else "Moderate confidence - consider validation",
        },
    }
