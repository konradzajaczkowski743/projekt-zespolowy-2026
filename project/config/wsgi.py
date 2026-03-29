"""
WSGI configuration for the the project project.

Exposes the WSGI callable as a module-level variable named ``application``.
Used for synchronous deployments with servers such as Gunicorn or uWSGI.

Documentation:
    https://docs.djangoproject.com/en/5.0/howto/deployment/wsgi/
"""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

application = get_wsgi_application()
