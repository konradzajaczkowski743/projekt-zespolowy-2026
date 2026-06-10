"""
REST API views for the project.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import urllib.parse
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen
import uuid
from typing import Any

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework import status, viewsets
from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.decorators import action

from api.models import AnalysisResult, ImageAnalysisRequest, ImageQuery
from api.serializers import (
    AnalysisRequestSerializer,
    AnalysisStatusSerializer,
    ImageQueryRequestSerializer,
    ImageQueryResponseSerializer,
)
from ml_engine.tasks import run_analysis_task, process_image_query_task

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
            uploaded_file = validated_data.get("image")
            file_size = None

            if validated_data.get("image_url"):
                image_url = validated_data["image_url"]
                try:
                    request_obj = UrlRequest(
                        image_url,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    with urlopen(request_obj, timeout=15) as response:
                        content_type = response.headers.get("Content-Type", "")
                        if not content_type.startswith("image/"):
                            raise ValueError("URL did not return an image content type.")
                        image_bytes = response.read()
                except (HTTPError, URLError, ValueError) as exc:
                    logger.warning("Failed to download image from URL %s: %s", image_url, exc)
                    return Response(
                        {"image_url": ["Nie można pobrać obrazu z podanego URL."]},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                file_name = os.path.basename(urllib.parse.urlparse(image_url).path) or "downloaded_image"
                if not os.path.splitext(file_name)[1]:
                    extension = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".jpg"
                    file_name += extension
                uploaded_file = ContentFile(image_bytes, name=file_name)
                file_size = len(image_bytes)

            # Truncate filename to avoid FileField max_length=100 limitation
            original_name = uploaded_file.name
            name_parts = os.path.splitext(original_name)
            base_name = name_parts[0][:50]  # Keep first 50 chars of basename
            extension = name_parts[1][:10]  # Keep extension
            safe_filename = base_name + extension
            
            destination_path: str = os.path.join(
                "photos", str(task_id), safe_filename
            )
            saved_path: str = default_storage.save(
                destination_path,
                ContentFile(uploaded_file.read()),
            )
            logger.debug("Saved uploaded image: path=%s (original=%s)", saved_path, original_name)

            analysis_request: ImageAnalysisRequest = ImageAnalysisRequest.objects.create(
                id=task_id,
                status=ImageAnalysisRequest.Status.PENDING,
                file=saved_path,
                original_filename=uploaded_file.name,
                file_size=file_size if file_size is not None else uploaded_file.size,
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


class ImageQueryViewSet(viewsets.ViewSet):
    """
    ViewSet for submitting queries about analyzed images.

    Endpoints:
        POST   /api/v1/queries/                    — Submit a new query about an image
        GET    /api/v1/queries/<query_id>/         — Get query status and response
    """

    parser_classes = [JSONParser]

    def create(self, request: Request) -> Response:
        """
        Submit a new query about an analyzed image.

        Args:
            request: DRF Request containing:
                - result_id (UUID): ID of the ImageAnalysisRequest (task_id)
                - query (str): The user's question about the image

        Returns:
            202 Accepted response with query_id.
        """
        try:
            # Get the result_id from the request (this is actually the task_id)
            result_id = request.data.get('result_id')
            if not result_id:
                return Response(
                    {"error": "result_id is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Validate the query
            serializer = ImageQueryRequestSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            # Check if the analysis request exists and has a result
            try:
                analysis_request = ImageAnalysisRequest.objects.get(id=result_id)
                analysis_result = analysis_request.result
            except ImageAnalysisRequest.DoesNotExist:
                return Response(
                    {"error": "Analysis request not found"},
                    status=status.HTTP_404_NOT_FOUND
                )
            except AnalysisResult.DoesNotExist:
                return Response(
                    {"error": "Analysis result not found - analysis may not be complete"},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Create the query
            query_id = uuid.uuid4()
            image_query = ImageQuery.objects.create(
                id=query_id,
                result=analysis_result,
                user_query=serializer.validated_data['query'],
                status=ImageQuery.Status.PENDING,
            )
            logger.info("Created ImageQuery: query_id=%s result_id=%s", query_id, result_id)

            # Dispatch the query processing task
            process_image_query_task.delay(str(query_id))
            logger.info("Dispatched query processing task for query_id=%s", query_id)

            response_serializer = ImageQueryResponseSerializer(image_query)
            return Response(
                response_serializer.data,
                status=status.HTTP_202_ACCEPTED,
            )

        except Exception as exc:
            logger.exception("Error creating query: %s", exc)
            return Response(
                {
                    "error": "Internal server error",
                    "details": str(exc),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def retrieve(self, request: Request, pk: str = None) -> Response:
        """
        Retrieve the status and response of a specific query.

        Args:
            request: DRF Request.
            pk: Query ID (UUID) string.

        Returns:
            The query status and AI response if available.
        """
        try:
            image_query = ImageQuery.objects.get(id=pk)
            serializer = ImageQueryResponseSerializer(image_query)
            return Response(serializer.data, status=status.HTTP_200_OK)

        except ImageQuery.DoesNotExist:
            return Response(
                {"error": "Query not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as exc:
            logger.exception("Error retrieving query %s: %s", pk, exc)
            return Response(
                {"error": "Internal server error", "details": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


def analysis_ui(request: Request) -> Response:
    """Display a simple upload page for frontend testing."""
    from django.shortcuts import render

    return render(request, "api/upload.html")
