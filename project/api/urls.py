"""
URL configuration for the project.

Routes:
    POST  /api/v1/analysis/          — Submit a new image-analysis job.
    GET   /api/v1/analysis/<task_id>/ — Poll status / retrieve results.
"""
from django.urls import path
from rest_framework.response import Response
from rest_framework.decorators import api_view

from api.views import AnalysisViewSet

# Use explicit path() definitions rather than a router so that UUID
# path converters provide type-safe matching out of the box.
analysis_create = AnalysisViewSet.as_view({"post": "create"})
analysis_detail = AnalysisViewSet.as_view({"get": "retrieve"})


@api_view(['GET'])
def api_root(request):
    """API root endpoint"""
    return Response({
        'message': 'Projekt API',
        'version': '1.0',
        'endpoints': {
            'analysis': '/api/v1/analysis/',
        }
    })


urlpatterns = [
    path("", api_root, name="api-root"),
    path("analysis/", analysis_create, name="analysis-create"),
    path("analysis/<uuid:pk>/", analysis_detail, name="analysis-detail"),
]
