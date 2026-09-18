"""Web interface and REST API for the recogniser."""

from __future__ import annotations

__all__ = ["create_app", "serve"]


def create_app(*args, **kwargs):
    """Build the FastAPI application. Imported lazily so the CLI stays light."""
    from .server import create_app as _create_app

    return _create_app(*args, **kwargs)


def serve(*args, **kwargs):
    from .server import serve as _serve

    return _serve(*args, **kwargs)
