"""
ASGI configuration for the the project project.

Exposes the ASGI callable as a module-level variable named ``application``.
Required if you want to serve the application using an ASGI server such as
Daphne or Uvicorn (needed for WebSocket / async Django support).

Documentation:
    https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

application = get_asgi_application()
