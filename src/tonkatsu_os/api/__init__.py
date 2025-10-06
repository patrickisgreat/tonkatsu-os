"""
FastAPI backend for Tonkatsu-OS.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .main import app as fastapi_app


def __getattr__(name: str) -> Any:
    if name == "app":
        from .main import app as fastapi_app

        return fastapi_app
    raise AttributeError(f"module 'tonkatsu_os.api' has no attribute {name!r}")


__all__ = ["app"]
