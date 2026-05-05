"""
the project Django configuration package.

This package is the main Django configuration module.
Celery app is imported here so that ``shared_task`` decorators
use the configured application instance.
"""
from .celery import app as celery_app

__all__ = ("celery_app",)
