"""
Pre-trained model management API routes.
"""

import logging
from typing import Dict, List

from fastapi import APIRouter, HTTPException

from ..models import ApiResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def get_pretrained_manager():
    """Dependency to get pre-trained model manager."""
    from tonkatsu_os.ml.pretrained_models import PreTrainedModelManager
    return PreTrainedModelManager()


@router.get("/models", response_model=Dict)
async def list_pretrained_models(manager=None):
    """List all available pre-trained models."""
    try:
        if manager is None:
            manager = get_pretrained_manager()
        
        models = manager.list_available_models()
        return {
            "available_models": models,
            "total_models": len(models),
            "loaded_models": list(manager.loaded_models.keys())
        }
        
    except Exception as e:
        logger.error(f"Error listing pre-trained models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/load", response_model=ApiResponse)
async def load_pretrained_model(model_name: str, manager=None):
    """Load a specific pre-trained model."""
    try:
        if manager is None:
            manager = get_pretrained_manager()
        
        success = manager.load_model(model_name)
        
        if success:
            return ApiResponse(
                success=True, 
                message=f"Successfully loaded model: {model_name}"
            )
        else:
            return ApiResponse(
                success=False, 
                error=f"Failed to load model: {model_name}"
            )
            
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/load-all", response_model=ApiResponse)
async def load_all_pretrained_models(manager=None):
    """Load all available pre-trained models."""
    try:
        if manager is None:
            manager = get_pretrained_manager()
        
        models = manager.list_available_models()
        loaded_count = 0
        failed_models = []
        
        for model_name in models.keys():
            try:
                if manager.load_model(model_name):
                    loaded_count += 1
                else:
                    failed_models.append(model_name)
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
                failed_models.append(model_name)
        
        message = f"Loaded {loaded_count}/{len(models)} models"
        if failed_models:
            message += f". Failed: {', '.join(failed_models)}"
        
        return ApiResponse(
            success=loaded_count > 0,
            message=message,
            data={
                "loaded_count": loaded_count,
                "total_models": len(models),
                "failed_models": failed_models
            }
        )
        
    except Exception as e:
        logger.error(f"Error loading all models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/status", response_model=Dict)
async def get_pretrained_models_status(manager=None):
    """Get status of all pre-trained models."""
    try:
        if manager is None:
            manager = get_pretrained_manager()
        
        available = manager.list_available_models()
        loaded = manager.loaded_models
        
        status = {}
        for model_name, model_info in available.items():
            status[model_name] = {
                **model_info,
                "loaded": model_name in loaded,
                "memory_usage": "unknown"  # Could add actual memory tracking
            }
        
        return {
            "models": status,
            "summary": {
                "total_available": len(available),
                "total_loaded": len(loaded),
                "memory_efficient": True  # Mock value
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        raise HTTPException(status_code=500, detail=str(e))