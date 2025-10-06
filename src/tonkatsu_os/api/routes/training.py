"""
ML model training API routes.
"""

import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ..models import ApiResponse, TrainingConfig, TrainingResult, TrainingStatus

router = APIRouter()
logger = logging.getLogger(__name__)

# Global training state
training_state = {
    "is_training": False,
    "progress": 0.0,
    "model_exists": False,
    "last_trained": None,
}


def get_database():
    """Dependency to get database instance."""
    from tonkatsu_os.database import RamanSpectralDatabase

    return RamanSpectralDatabase()


def get_classifier():
    """Dependency to get classifier instance."""
    from tonkatsu_os.ml import EnsembleClassifier

    return EnsembleClassifier()


@router.post("/train", response_model=TrainingResult)
async def train_models(
    config: Optional[TrainingConfig] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db=Depends(get_database),
    classifier=Depends(get_classifier),
):
    """
    Train machine learning models on the spectral database.

    This trains ensemble ML models (Random Forest, SVM, Neural Network)
    on all available spectral data in the database.
    """
    try:
        if training_state["is_training"]:
            raise HTTPException(status_code=409, detail="Training already in progress")

        # Check if we have enough data
        stats = db.get_database_stats()
        if stats["total_spectra"] < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 10 spectra for training. Current: {stats['total_spectra']}",
            )

        # Use default config if none provided
        if config is None:
            config = TrainingConfig()

        # Start REAL training in background
        background_tasks.add_task(_train_models_background, config, db, classifier)

        training_state["is_training"] = True
        training_state["progress"] = 0.0

        # Return immediate response - training runs in background
        return TrainingResult(
            rf_accuracy=0.0,  # Will be updated when training completes
            svm_accuracy=0.0,
            nn_accuracy=0.0,
            ensemble_accuracy=0.0,
            n_train_samples=0,
            n_val_samples=0,
            n_features=0,
            n_classes=0,
            classification_report={},
            confusion_matrix=[],
            training_time=0.0
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status."""
    try:
        import os
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "..", "..", "..", "trained_ensemble_model.pkl")
        model_path = os.path.abspath(model_path)
        model_exists = os.path.exists(model_path)

        return TrainingStatus(
            is_training=training_state["is_training"],
            progress=training_state["progress"] if training_state["is_training"] else None,
            model_exists=model_exists,
            last_trained=training_state["last_trained"],
        )

    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=dict)
async def get_model_metrics():
    """Get performance metrics for trained models."""
    try:
        model_path = "trained_model.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="No trained model found")

        # Return mock metrics
        # In production, this would load actual model metrics
        return {
            "ensemble_accuracy": 0.87,
            "random_forest_accuracy": 0.84,
            "svm_accuracy": 0.82,
            "neural_network_accuracy": 0.89,
            "training_samples": 150,
            "validation_samples": 38,
            "n_features": 67,
            "n_classes": 8,
            "training_time": 45.2,
            "cross_validation_score": 0.85,
            "feature_importance": {
                "spectral_centroid": 0.23,
                "num_peaks": 0.18,
                "dominant_peak_intensity": 0.15,
                "spectral_spread": 0.12,
                "peak_density": 0.10,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=ApiResponse)
async def stop_training():
    """Stop current training process."""
    try:
        if not training_state["is_training"]:
            return ApiResponse(success=False, message="No training in progress")

        training_state["is_training"] = False
        training_state["progress"] = 0.0

        return ApiResponse(success=True, message="Training stopped")

    except Exception as e:
        logger.error(f"Error stopping training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/model", response_model=ApiResponse)
async def delete_model():
    """Delete trained model."""
    try:
        model_path = "trained_model.pkl"
        if os.path.exists(model_path):
            os.remove(model_path)
            training_state["model_exists"] = False
            training_state["last_trained"] = None

            return ApiResponse(success=True, message="Model deleted successfully")
        else:
            return ApiResponse(success=False, message="No model to delete")

    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _train_models_background(config: TrainingConfig, db, classifier):
    """Background task for REAL model training."""
    try:
        import numpy as np
        from datetime import datetime
        from tonkatsu_os.preprocessing import AdvancedPreprocessor

        logger.info("Starting REAL model training...")
        
        # Step 1: Load all spectra from database (20% progress)
        training_state["progress"] = 0.1
        logger.info("Loading spectra from database...")
        
        spectra_data = _load_training_data_from_database(db)
        if len(spectra_data) < 10:
            raise ValueError(f"Need at least 10 spectra for training. Found: {len(spectra_data)}")
        
        training_state["progress"] = 0.2
        logger.info(f"Loaded {len(spectra_data)} spectra from database")
        
        # Step 2: Extract features from all spectra (40% progress)
        training_state["progress"] = 0.3
        logger.info("Extracting spectral features...")
        
        X, y, feature_names = _extract_features_for_training(spectra_data)
        
        training_state["progress"] = 0.4
        logger.info(f"Extracted {X.shape[1]} features from {X.shape[0]} spectra")
        
        # Step 3: Configure classifier with user settings
        training_state["progress"] = 0.5
        logger.info("Configuring ensemble classifier...")
        
        classifier = _configure_classifier(config)
        
        # Step 4: Train the ensemble model (60-90% progress)
        training_state["progress"] = 0.6
        logger.info("Training ensemble models...")
        
        training_results = classifier.train(X, y, validation_split=config.validation_split)
        
        training_state["progress"] = 0.9
        logger.info("Training completed, saving model...")
        
        # Step 5: Save trained model (100% progress)
        import os
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "..", "..", "..", "trained_ensemble_model.pkl")
        model_path = os.path.abspath(model_path)
        classifier.save_model(model_path)
        
        training_state["progress"] = 1.0
        training_state["is_training"] = False
        training_state["model_exists"] = True
        training_state["last_trained"] = datetime.now()
        
        logger.info(f"Model training completed successfully! Saved to {model_path}")
        logger.info(f"Training results: {training_results}")
        
        return training_results

    except Exception as e:
        logger.error(f"REAL training failed: {e}")
        training_state["is_training"] = False
        training_state["progress"] = 0.0
        raise


def _load_training_data_from_database(db) -> list:
    """Load all spectra from database for training."""
    try:
        # Get all spectra from database
        stats = db.get_database_stats()
        logger.info(f"Database contains {stats['total_spectra']} total spectra")
        
        # Load all spectra data
        # This should be implemented in the database class
        all_spectra = []
        
        # For now, use search to get all compounds
        for compound_name, count in stats.get('top_compounds', []):
            compound_spectra = db.search_by_compound_name(compound_name, exact_match=True)
            for spectrum_basic in compound_spectra:
                # Get full spectrum data including spectrum_data array
                full_spectrum = db.get_spectrum_by_id(spectrum_basic['id'])
                if full_spectrum and full_spectrum.get('spectrum_data') is not None:
                    all_spectra.append(full_spectrum)
        
        logger.info(f"Successfully loaded {len(all_spectra)} spectra for training")
        return all_spectra
        
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise


def _extract_features_for_training(spectra_data: list):
    """Extract ML features from spectral data."""
    try:
        from tonkatsu_os.preprocessing import AdvancedPreprocessor
        import numpy as np
        
        preprocessor = AdvancedPreprocessor()
        
        features_list = []
        labels = []
        
        for spectrum_data in spectra_data:
            try:
                # Get spectrum array
                spectrum_array = np.array(spectrum_data['spectrum_data'])
                compound_name = spectrum_data['compound_name']
                
                # Preprocess spectrum
                processed_spectrum = preprocessor.preprocess(spectrum_array)
                
                # Extract spectral features
                features = preprocessor.spectral_features(processed_spectrum)
                
                # Convert features dict to array
                feature_vector = []
                feature_vector.extend([
                    features.get('spectral_centroid', 0),
                    features.get('spectral_spread', 0),
                    features.get('spectral_skewness', 0),
                    features.get('spectral_kurtosis', 0),
                    features.get('spectral_rolloff', 0),
                    features.get('spectral_flatness', 0),
                    features.get('zero_crossing_rate', 0),
                    features.get('num_peaks', 0),
                    features.get('peak_density', 0),
                    features.get('dominant_peak_intensity', 0),
                    features.get('mean_intensity', 0),
                    features.get('std_intensity', 0),
                    features.get('max_intensity', 0),
                    features.get('min_intensity', 0)
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
                
                # Add spectral bins (simplified - take every 50th point)
                spectrum_bins = processed_spectrum[::50]  # Sample spectrum
                feature_vector.extend(spectrum_bins.tolist())
                
                features_list.append(feature_vector)
                labels.append(compound_name)
                
            except Exception as e:
                logger.warning(f"Failed to extract features for spectrum {spectrum_data.get('id', 'unknown')}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No valid features extracted from spectra")
        
        X = np.array(features_list)
        y = np.array(labels)
        
        # Feature names for interpretability
        feature_names = [
            'spectral_centroid', 'spectral_spread', 'spectral_skewness', 'spectral_kurtosis',
            'spectral_rolloff', 'spectral_flatness', 'zero_crossing_rate', 'num_peaks',
            'peak_density', 'dominant_peak_intensity', 'mean_intensity', 'std_intensity',
            'max_intensity', 'min_intensity', 'mean_spectrum', 'std_spectrum', 'median_spectrum',
            'q25_spectrum', 'q75_spectrum', 'max_spectrum', 'min_spectrum'
        ]
        
        # Add spectral bin names
        bin_names = [f'spectrum_bin_{i}' for i in range(X.shape[1] - len(feature_names))]
        feature_names.extend(bin_names)
        
        logger.info(f"Extracted {X.shape[1]} features from {X.shape[0]} spectra")
        logger.info(f"Unique compounds: {len(np.unique(y))}")
        
        return X, y, feature_names
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise


def _configure_classifier(config: TrainingConfig):
    """Configure ensemble classifier with user settings."""
    from tonkatsu_os.ml import EnsembleClassifier
    
    classifier = EnsembleClassifier(
        use_pca=config.use_pca,
        n_components=config.n_components,
        use_pls=True  # Always include PLS for spectroscopic data
    )
    
    logger.info(f"Configured classifier: PCA={config.use_pca}, n_components={config.n_components}")
    return classifier


def _create_mock_training_result() -> dict:
    """Create mock training results."""
    return {
        "rf_accuracy": 0.84,
        "svm_accuracy": 0.82,
        "nn_accuracy": 0.89,
        "ensemble_accuracy": 0.87,
        "n_train_samples": 150,
        "n_val_samples": 38,
        "n_features": 67,
        "n_classes": 8,
        "classification_report": {
            "benzene": {"precision": 0.89, "recall": 0.85, "f1-score": 0.87},
            "ethanol": {"precision": 0.92, "recall": 0.88, "f1-score": 0.90},
            "water": {"precision": 0.95, "recall": 0.93, "f1-score": 0.94},
        },
        "confusion_matrix": [[25, 2, 1], [1, 22, 0], [0, 1, 26]],
        "training_time": 45.2,
    }
