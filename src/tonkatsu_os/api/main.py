"""
FastAPI main application for Tonkatsu-OS backend.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from tonkatsu_os import __version__
from tonkatsu_os.hardware import HardwareManager

from .models import SystemHealth
from .state import app_state
from .routes import acquisition, analysis, database, import_data, pretrained, reference, system, training

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip('"\'')
                        os.environ[key.strip()] = value
            logger.info("Loaded environment variables from .env file")
        except Exception as e:
            logger.warning(f"Failed to load .env file: {e}")

# Load .env file on startup
load_env_file()

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Tonkatsu-OS API server...")

    # Initialize database and other components
    try:
        from tonkatsu_os.database import RamanSpectralDatabase
        from tonkatsu_os.ml import EnsembleClassifier
        from tonkatsu_os.preprocessing import AdvancedPreprocessor

        app_state["database"] = RamanSpectralDatabase()
        app_state["preprocessor"] = AdvancedPreprocessor()
        app_state["hardware_manager"] = HardwareManager()

        classifier = EnsembleClassifier()
        model_path = Path("trained_ensemble_model.pkl")
        metrics_path = Path("trained_ensemble_metrics.json")

        if model_path.exists():
            try:
                classifier.load_model(str(model_path))
                logger.info("Loaded trained ensemble model from %s", model_path)
                if metrics_path.exists():
                    try:
                        app_state["model_metrics"] = json.loads(metrics_path.read_text())
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse metrics file %s", metrics_path)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to load existing model %s: %s", model_path, exc)

        app_state["classifier"] = classifier

        logger.info("Components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down Tonkatsu-OS API server...")
    if "database" in app_state:
        app_state["database"].close()
    if "hardware_manager" in app_state:
        try:
            app_state["hardware_manager"].disconnect_spectrometer()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to disconnect spectrometer on shutdown: %s", exc)


# Create FastAPI app
app = FastAPI(
    title="Tonkatsu-OS API",
    description="AI-Powered Raman Molecular Identification Platform",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000", 
        "http://localhost:3001", 
        "http://127.0.0.1:3001"
    ],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "status_code": 500},
    )


# Include routers
app.include_router(database.router, prefix="/api/database", tags=["database"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(import_data.router, prefix="/api/import", tags=["import"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(pretrained.router, prefix="/api/pretrained", tags=["pretrained"])
app.include_router(reference.router, prefix="/api/reference", tags=["reference"])
app.include_router(acquisition.router, prefix="/api/acquisition", tags=["acquisition"])
app.include_router(system.router, prefix="/api/system", tags=["system"])


# Root endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Tonkatsu-OS API",
        "version": __version__,
        "description": "AI-Powered Raman Molecular Identification Platform",
        "docs_url": "/docs",
        "health_check": "/api/system/health",
    }


@app.get("/api/health")
async def health_check():
    """Simple health check endpoint."""
    try:
        # Check database connection
        db = app_state.get("database")
        if db:
            stats = db.get_database_stats()
            db_healthy = True
        else:
            db_healthy = False

        # Check other components
        preprocessor_healthy = app_state.get("preprocessor") is not None
        classifier_healthy = app_state.get("classifier") is not None

        status = (
            "healthy" if all([db_healthy, preprocessor_healthy, classifier_healthy]) else "warning"
        )

        return SystemHealth(
            status=status,
            components={
                "database": db_healthy,
                "preprocessor": preprocessor_healthy,
                "classifier": classifier_healthy,
            },
            version=__version__,
            uptime=None,
            memory_usage=None,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return SystemHealth(
            status="error",
            components={
                "database": False,
                "preprocessor": False,
                "classifier": False,
            },
            version=__version__,
            uptime=None,
            memory_usage=None,
        )


# Dependency injection
def get_database():
    """Get database instance."""
    db = app_state.get("database")
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return db


def get_preprocessor():
    """Get preprocessor instance."""
    preprocessor = app_state.get("preprocessor")
    if not preprocessor:
        raise HTTPException(status_code=500, detail="Preprocessor not initialized")
    return preprocessor


def get_classifier():
    """Get classifier instance."""
    classifier = app_state.get("classifier")
    if not classifier:
        raise HTTPException(status_code=500, detail="Classifier not initialized")
    return classifier


# Make dependencies available to routes
app.dependency_overrides = {}
app.get_database = get_database
app.get_preprocessor = get_preprocessor
app.get_classifier = get_classifier

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
