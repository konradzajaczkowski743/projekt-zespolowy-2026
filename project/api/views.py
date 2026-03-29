"""
REST API views for the project.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework import status, viewsets
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response

from api.models import AnalysisResult, ImageAnalysisRequest
from api.serializers import AnalysisRequestSerializer, AnalysisStatusSerializer
from ml_engine.pipeline import run_full_analysis

logger: logging.Logger = logging.getLogger(__name__)


class AnalysisViewSet(viewsets.ViewSet):
    """
    ViewSet for submitting image-analysis jobs and polling their status.

    Endpoints:
        POST   /api/v1/analysis/   
    """

    parser_classes = [MultiPartParser, FormParser]

    def create(self, request: Request) -> Response:
        """
        Submit a new image-analysis job.

        Validates the multipart request, saves uploaded images to the media
        store, persists an ``ImageAnalysisRequest`` record, and executes
        the ML processing pipeline synchronously.

        Args:
            request: DRF ``Request`` containing ``images`` file list and
                optional ``latitude`` / ``longitude`` fields.

        Returns:
            ``202 Accepted`` response with a ``task_id`` and initial status.

        Response schema::

            {
                "task_id": "<uuid>",
                "status": "PENDING",
                "created_at": "<iso8601>",
                "updated_at": "<iso8601>",
                "result": null
            }
        """
        serializer = AnalysisRequestSerializer(data=request.data)
        if not serializer.is_valid():
            logger.warning(
                "Analysis request validation failed. errors=%s",
                serializer.errors,
            )
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_data: dict[str, Any] = serializer.validated_data

        task_id: uuid.UUID = uuid.uuid4()
        image_paths: list[str] = []

        for uploaded_file in validated_data["images"]:
            destination_path: str = os.path.join(
                "photos", str(task_id), uploaded_file.name
            )
            saved_path: str = default_storage.save(
                destination_path,
                ContentFile(uploaded_file.read()),
            )
            image_paths.append(saved_path)
            logger.debug("Saved uploaded image: path=%s", saved_path)

        analysis_request: ImageAnalysisRequest = ImageAnalysisRequest.objects.create(
            id=task_id,
            status=ImageAnalysisRequest.Status.PENDING,
            images=image_paths,
            latitude=validated_data.get("latitude"),
            longitude=validated_data.get("longitude"),
        )
        logger.info("Created AnalysisRequest: task_id=%s", task_id)

        try:
            run_full_analysis(str(task_id))
            logger.info("Executed ML pipeline for task_id=%s", task_id)
        except Exception as e:
            logger.error("ML pipeline failed for task_id=%s: %s", task_id, e)

        analysis_request.refresh_from_db()

        response_serializer = AnalysisStatusSerializer(analysis_request)
        return Response(
            response_serializer.data,
            status=status.HTTP_200_OK,
        )
