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

        # Start training in background
        background_tasks.add_task(_train_models_background, config, db, classifier)

        training_state["is_training"] = True
        training_state["progress"] = 0.0

        # Return mock training results for now
        # In production, this would wait for actual training to complete
        mock_result = _create_mock_training_result()

        return TrainingResult(**mock_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status."""
    try:
        model_path = "trained_model.pkl"
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
    """Background task for model training."""
    try:
        import time
        from datetime import datetime

        logger.info("Starting background model training...")

        # Simulate training progress
        for i in range(10):
            time.sleep(1)  # Simulate work
            training_state["progress"] = (i + 1) / 10
            logger.info(f"Training progress: {training_state['progress']:.1%}")

        # In production, this would do actual training:
        # 1. Load data from database
        # 2. Extract features
        # 3. Train ensemble models
        # 4. Save trained model

        training_state["is_training"] = False
        training_state["progress"] = 1.0
        training_state["model_exists"] = True
        training_state["last_trained"] = datetime.now()

        logger.info("Background training completed")

    except Exception as e:
        logger.error(f"Background training failed: {e}")
        training_state["is_training"] = False
        training_state["progress"] = 0.0


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
