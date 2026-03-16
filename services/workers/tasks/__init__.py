"""Celery tasks package for async processing."""

from .celery_app import celery_app

__all__ = ["celery_app"]
