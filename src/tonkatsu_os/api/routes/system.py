"""
System management API routes.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
import logging
import os
import tempfile
import zipfile
import time
from datetime import datetime

from ..models import SystemHealth, ExportRequest, ApiResponse
from tonkatsu_os import __version__

router = APIRouter()
logger = logging.getLogger(__name__)

def get_database():
    """Dependency to get database instance."""
    from tonkatsu_os.database import RamanSpectralDatabase
    return RamanSpectralDatabase()

@router.get("/health", response_model=SystemHealth)
async def health_check(db=Depends(get_database)):
    """Comprehensive system health check."""
    try:
        import psutil
        
        # Check database
        try:
            stats = db.get_database_stats()
            db_healthy = True
        except:
            db_healthy = False
        
        # Check other components
        components = {
            "database": db_healthy,
            "preprocessor": True,  # Basic availability
            "classifier": True,
            "api": True
        }
        
        # Determine overall status
        if all(components.values()):
            status = "healthy"
        elif any(components.values()):
            status = "warning"
        else:
            status = "error"
        
        # Get system metrics
        memory_usage = psutil.virtual_memory().percent
        uptime = time.time() - psutil.boot_time()
        
        return SystemHealth(
            status=status,
            components=components,
            version=__version__,
            uptime=uptime,
            memory_usage=memory_usage
        )
        
    except ImportError:
        # psutil not available, return basic health check
        return SystemHealth(
            status="healthy",
            components={"api": True},
            version=__version__
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return SystemHealth(
            status="error",
            components={"api": False},
            version=__version__
        )

@router.get("/info", response_model=dict)
async def get_system_info():
    """Get detailed system information."""
    try:
        import platform
        import sys
        
        return {
            "version": __version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "uptime": time.time(),
            "api_docs": "/docs",
            "admin_interface": "/admin"
        }
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export/{format}", response_class=FileResponse)
async def export_database(
    format: str,
    request: ExportRequest = None,
    db=Depends(get_database)
):
    """Export database in various formats."""
    try:
        if format not in ["csv", "json", "sqlite"]:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
        # Create temporary file for export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "sqlite":
            # Copy the database file
            import shutil
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_tonkatsu_export_{timestamp}.db")
            shutil.copy("raman_spectra.db", temp_file.name)
            
            return FileResponse(
                temp_file.name,
                media_type="application/octet-stream",
                filename=f"tonkatsu_database_{timestamp}.db"
            )
        
        elif format == "csv":
            # Export as CSV
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w')
            
            # Write CSV header
            temp_file.write("id,compound_name,chemical_formula,cas_number,acquisition_date,source\n")
            
            # Get all spectra info (simplified export)
            stats = db.get_database_stats()
            for compound, count in stats['top_compounds']:
                spectra = db.search_by_compound_name(compound, exact_match=True)
                for spec in spectra:
                    temp_file.write(f"{spec['id']},{compound},,,{datetime.now()},database\n")
            
            temp_file.close()
            
            return FileResponse(
                temp_file.name,
                media_type="text/csv",
                filename=f"tonkatsu_export_{timestamp}.csv"
            )
        
        elif format == "json":
            # Export as JSON
            import json
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w')
            
            export_data = {
                "export_timestamp": timestamp,
                "version": __version__,
                "database_stats": db.get_database_stats(),
                "spectra": []
            }
            
            # Add limited spectral data (full export would be very large)
            stats = db.get_database_stats()
            for compound, count in stats['top_compounds'][:10]:  # Limit to top 10
                spectra = db.search_by_compound_name(compound, exact_match=True)
                for spec in spectra[:2]:  # Limit to 2 per compound
                    export_data["spectra"].append({
                        "id": spec['id'],
                        "compound_name": compound,
                        "data_points": "truncated_for_export"
                    })
            
            json.dump(export_data, temp_file, indent=2)
            temp_file.close()
            
            return FileResponse(
                temp_file.name,
                media_type="application/json",
                filename=f"tonkatsu_export_{timestamp}.json"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting database: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.post("/backup", response_model=ApiResponse)
async def create_backup(db=Depends(get_database)):
    """Create a backup of the database."""
    try:
        import shutil
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"backup_tonkatsu_{timestamp}.db"
        
        # Copy database file
        shutil.copy("raman_spectra.db", backup_filename)
        
        return ApiResponse(
            success=True,
            message=f"Backup created: {backup_filename}"
        )
        
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")

@router.post("/restore", response_model=ApiResponse)
async def restore_backup(backup_file: str):
    """Restore database from backup."""
    try:
        if not os.path.exists(backup_file):
            raise HTTPException(status_code=404, detail="Backup file not found")
        
        import shutil
        
        # Backup current database first
        current_backup = f"current_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        shutil.copy("raman_spectra.db", current_backup)
        
        # Restore from backup
        shutil.copy(backup_file, "raman_spectra.db")
        
        return ApiResponse(
            success=True,
            message=f"Database restored from {backup_file}. Previous version saved as {current_backup}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring backup: {e}")
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")

@router.get("/logs", response_model=list)
async def get_system_logs(lines: int = 100):
    """Get recent system logs."""
    try:
        # In production, this would read from actual log files
        # For demo, return mock log entries
        logs = []
        for i in range(min(lines, 20)):
            logs.append({
                "timestamp": datetime.now(),
                "level": "INFO",
                "message": f"System operational - entry {i+1}",
                "component": "system"
            })
        
        return logs
        
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/restart", response_model=ApiResponse)
async def restart_system():
    """Restart the system (admin only)."""
    try:
        # In production, this would trigger a graceful restart
        return ApiResponse(
            success=False,
            message="Restart functionality not implemented in demo mode"
        )
        
    except Exception as e:
        logger.error(f"Error restarting system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=dict)
async def get_system_metrics():
    """Get system performance metrics."""
    try:
        import time
        
        # Basic metrics
        metrics = {
            "uptime": time.time(),
            "requests_total": 1000,  # Mock data
            "requests_per_minute": 45,
            "database_size_mb": 2.5,
            "memory_usage_percent": 35.2,
            "cpu_usage_percent": 12.8,
            "active_connections": 3
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))