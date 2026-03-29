"""
URL configuration for the project.

Routes:
    POST  /api/v1/analysis/          — Submit a new image-analysis job.
    GET   /api/v1/analysis/<task_id>/ — Poll status / retrieve results.
"""
from django.urls import path

from api.views import AnalysisViewSet

# Use explicit path() definitions rather than a router so that UUID
# path converters provide type-safe matching out of the box.
analysis_create = AnalysisViewSet.as_view({"post": "create"})

urlpatterns = [
    path("analysis/", analysis_create, name="analysis-create"),
]
