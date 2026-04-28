from django.urls import include, path
from django.conf import settings
from django.conf.urls.static import static
from api.views import analysis_ui

urlpatterns = [
    path("", analysis_ui, name="analysis-ui"),
    path("api/v1/", include("api.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
