"""ML model training API routes."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from tonkatsu_os.ml import EnsembleClassifier
from tonkatsu_os.ml.training_pipeline import (
    MODEL_PATH,
    METRICS_PATH,
    train_ensemble_model,
)

from ..models import ApiResponse, TrainingConfig, TrainingResult, TrainingStatus
from ..state import app_state

router = APIRouter()
logger = logging.getLogger(__name__)

# Global training state
training_state = {
    "is_training": False,
    "progress": 0.0,
    "model_exists": False,
    "last_trained": None,
}


@router.post("/train", response_model=TrainingResult)
async def train_models(config: Optional[TrainingConfig] = None):
    """
    Train machine learning models on the spectral database.

    This trains ensemble ML models (Random Forest, SVM, Neural Network)
    on all available spectral data in the database.
    """
    try:
        if training_state["is_training"]:
            raise HTTPException(status_code=409, detail="Training already in progress")

        if config is None:
            config = TrainingConfig()

        training_state["is_training"] = True
        training_state["progress"] = 0.1

        metrics = train_ensemble_model(
            use_pca=config.use_pca,
            use_pls=False,
            n_components=config.n_components,
            validation_split=config.validation_split,
            min_samples_per_class=3,
            rebuild_features=True,
        )

        training_state.update(
            {
                "is_training": False,
                "progress": 0.0,
                "model_exists": True,
                "last_trained": datetime.utcnow(),
            }
        )

        classifier = app_state.get("classifier") or EnsembleClassifier()
        classifier.load_model(str(MODEL_PATH))
        app_state["classifier"] = classifier
        app_state["model_metrics"] = metrics

        return TrainingResult(
            rf_accuracy=metrics.get("rf_accuracy", 0.0),
            svm_accuracy=metrics.get("svm_accuracy", 0.0),
            nn_accuracy=metrics.get("nn_accuracy", 0.0),
            pls_accuracy=metrics.get("pls_accuracy"),
            ensemble_accuracy=metrics.get("ensemble_accuracy", 0.0),
            n_train_samples=metrics.get("n_train_samples", 0),
            n_val_samples=metrics.get("n_val_samples", 0),
            n_features=metrics.get("n_features", 0),
            n_classes=metrics.get("n_classes", 0),
            classification_report=metrics.get("classification_report", {}),
            confusion_matrix=metrics.get("confusion_matrix", []),
            training_time=metrics.get("training_time", 0.0),
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
        model_exists = MODEL_PATH.exists()

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
        if METRICS_PATH.exists():
            return json.loads(METRICS_PATH.read_text())

        metrics = app_state.get("model_metrics")
        if metrics:
            return metrics

        raise HTTPException(status_code=404, detail="No trained model found")

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
        removed = False
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
            removed = True
        if METRICS_PATH.exists():
            METRICS_PATH.unlink()
        training_state.update({"model_exists": False, "last_trained": None})
        app_state.pop("model_metrics", None)

        if removed:
            return ApiResponse(success=True, message="Model deleted successfully")
        return ApiResponse(success=False, message="No model to delete")

    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _train_models_background(*_args, **_kwargs):
    raise NotImplementedError("Background training is no longer supported")
