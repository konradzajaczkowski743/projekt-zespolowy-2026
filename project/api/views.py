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
from ml_engine.tasks import run_analysis_task

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

        Validates the multipart request, saves the uploaded image to the media
        store, persists an ``ImageAnalysisRequest`` record, and dispatches
        the ML processing pipeline asynchronously via Celery.

        Args:
            request: DRF ``Request`` containing ``image`` file and
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
        try:
            serializer = AnalysisRequestSerializer(data=request.data)
            if not serializer.is_valid():
                logger.warning(
                    "Analysis request validation failed. errors=%s",
                    serializer.errors,
                )
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            validated_data: dict[str, Any] = serializer.validated_data

            task_id: uuid.UUID = uuid.uuid4()
            uploaded_file = validated_data["image"]
            destination_path: str = os.path.join(
                "photos", str(task_id), uploaded_file.name
            )
            saved_path: str = default_storage.save(
                destination_path,
                ContentFile(uploaded_file.read()),
            )
            logger.debug("Saved uploaded image: path=%s", saved_path)

            analysis_request: ImageAnalysisRequest = ImageAnalysisRequest.objects.create(
                id=task_id,
                status=ImageAnalysisRequest.Status.PENDING,
                file=saved_path,
                original_filename=uploaded_file.name,
                file_size=uploaded_file.size,
                latitude=validated_data.get("latitude"),
                longitude=validated_data.get("longitude"),
            )
            logger.info("Created AnalysisRequest: task_id=%s", task_id)

            # Dispatch ML pipeline asynchronously via Celery
            run_analysis_task.delay(str(task_id))
            logger.info("Dispatched async ML pipeline for task_id=%s", task_id)

            response_serializer = AnalysisStatusSerializer(analysis_request)
            return Response(
                response_serializer.data,
                status=status.HTTP_202_ACCEPTED,
            )
        except Exception as exc:
            logger.exception("Unhandled error in analysis create: %s", exc)
            return Response(
                {
                    "error": "Internal server error",
                    "details": str(exc),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def retrieve(self, request: Request, pk: str = None) -> Response:
        """
        Retrieve the status and results of a specific analysis job.

        Args:
            request: DRF Request.
            pk: Task ID (UUID) string.

        Returns:
            If completed, returns the full JSON result.
            Otherwise, returns the current status (e.g. {"status": "PROCESSING"}).
        """
        try:
            analysis_request = ImageAnalysisRequest.objects.get(id=pk)
            
            # If the job is completed, fetch and return the full result payload
            if analysis_request.status == ImageAnalysisRequest.Status.COMPLETED:
                try:
                    result = analysis_request.result
                    data = {
                        "task_id": str(analysis_request.id),
                        "status": analysis_request.status,
                        "forensics": result.forensics,
                        "image_description": result.image_description,
                        "geo_verification": result.geo_verification,
                        "objects_detected": result.objects_detected,
                        "distance_estimation": result.distance_estimation,
                        "alpr": result.alpr,
                        "processing_time_ms": result.processing_time_ms,
                    }
                    return Response(data, status=status.HTTP_200_OK)
                except AnalysisResult.DoesNotExist:
                    # In case the status is COMPLETED but result is missing
                    pass
            
            # For all other statuses, return the basic status object
            serializer = AnalysisStatusSerializer(analysis_request)
            return Response(serializer.data, status=status.HTTP_200_OK)
            
        except ImageAnalysisRequest.DoesNotExist:
            return Response(
                {"error": "Task not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as exc:
            logger.exception("Error retrieving analysis %s: %s", pk, exc)
            return Response(
                {"error": "Internal server error", "details": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


def analysis_ui(request: Request) -> Response:
    """Display a simple upload page for frontend testing."""
    from django.shortcuts import render

    return render(request, "api/upload.html")
