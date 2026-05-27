"""
Celery application configuration for the project.

Initialises a Celery app that reads its settings from Django's
``settings.py`` under the ``CELERY_`` namespace and auto-discovers
task modules in each installed Django application.

Usage::

    # In any app, create a ``tasks.py`` with::
    from celery import shared_task

    @shared_task
    def my_task(arg):
        ...

Workers are started via::

    celery -A config worker --loglevel=info
"""
from __future__ import annotations

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")

# Read Celery-related settings from Django settings.py.
# All settings prefixed with ``CELERY_`` will be picked up automatically.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discover tasks.py in all installed Django apps.
app.autodiscover_tasks()
