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
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response

from api.models import AnalysisResult, ImageAnalysisRequest
from api.serializers import AnalysisRequestSerializer, AnalysisStatusSerializer
from ml_engine.tasks import run_analysis_task

logger: logging.Logger = logging.getLogger(__name__)


def _generate_summary(result: AnalysisResult) -> str:
    """Generates a human-readable summary in Polish for the verdict section."""
    is_auth = result.forensics.get("is_authentic", True)
    score = result.forensics.get("confidence_score", 0.0)
    
    if is_auth:
        auth_text = f"Zdjęcie jest prawdopodobnie autentyczne (pewność: {score:.1f}%)."
    else:
        auth_text = f"UWAGA: Wykryto potencjalną manipulację lub generację AI (pewność autentyczności: {score:.1f}%)."
        
    location = result.geo_verification.get("predicted_region")
    if location:
        source = result.geo_verification.get("prediction_source", "nieznane")
        geo_conf = result.geo_verification.get("confidence_score", 0.0) * 100
        loc_text = f" Rozpoznano lokalizację: {location} (źródło: {source}, pewność: {geo_conf:.1f}%)."
    else:
        loc_text = " Nie udało się jednoznacznie rozpoznać lokalizacji."
        
    return auth_text + loc_text


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
                        "processing_time_ms": result.processing_time_ms,
                        "verdict": {
                            "is_authentic": result.forensics.get("is_authentic", True),
                            "authenticity_score": result.forensics.get("confidence_score", 0.0),
                            "predicted_location": result.geo_verification.get("predicted_region"),
                            "location_source": result.geo_verification.get("prediction_source"),
                            "location_confidence": result.geo_verification.get("confidence_score"),
                            "is_location_consistent": result.geo_verification.get("is_location_consistent", False),
                            "summary": _generate_summary(result),
                        },
                        "authenticity": result.forensics,
                        "location": {
                            "predicted_region": result.geo_verification.get("predicted_region"),
                            "prediction_source": result.geo_verification.get("prediction_source"),
                            "confidence": result.geo_verification.get("confidence_score"),
                            "is_location_consistent": result.geo_verification.get("is_location_consistent", False),
                            "distance_km": result.geo_verification.get("distance_km"),
                            "streetclip_top5": result.geo_verification.get("streetclip_top5", []),
                            "vit_predicted_region": result.geo_verification.get("vit_predicted_region"),
                        },
                        "content": {
                            "description": result.image_description.get("description", ""),
                            "scenes": result.image_description.get("scenes", []),
                            "detected_objects": result.objects_detected,
                            "described_objects": result.image_description.get("objects_identified", []),
                            "main_landmark": result.distance_estimation.get("main_object"),
                            "license_plates": result.alpr,
                        },
                        "metadata": {
                            "models_used": {
                                "forensics": "1.0.0-advanced (ELA + EXIF + ML ensemble)",
                                "geolocation": "StreetCLIP + Vector DB",
                                "description": result.image_description.get("metadata", {}).get("model_used", "gemma4:e2b"),
                                "object_detection": "rtdetr_r50vd",
                                "depth_estimation": result.distance_estimation.get("model_used", "depth-anything-v2-base")
                            }
                        }
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

    @action(detail=True, methods=['post'])
    def chat(self, request: Request, pk: str = None) -> Response:
        """
        Interactive VQA (Visual Question Answering) endpoint.
        
        Allows the user to ask questions about an image that has completed analysis.
        Uses Gemma4 model with the analysis result as context.
        """
        message = request.data.get("message")
        if not message:
            return Response({"error": "Missing 'message' in request body"}, status=status.HTTP_400_BAD_request)
            
        try:
            analysis_request = ImageAnalysisRequest.objects.get(id=pk)
            
            if analysis_request.status != ImageAnalysisRequest.Status.COMPLETED:
                return Response(
                    {"error": "Analysis is not completed yet."}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
                
            try:
                result = analysis_request.result
            except AnalysisResult.DoesNotExist:
                return Response({"error": "Result missing"}, status=status.HTTP_404_NOT_FOUND)
                
            # Prepare context for the LLM
            summary = _generate_summary(result)
            objects = ", ".join([d["label"] for d in result.objects_detected])
            plates = ", ".join([p["plate_text"] for p in result.alpr])
            
            context = (
                f"You are a helpful AI assistant answering questions about an image.\n"
                f"Image Analysis Context:\n"
                f"- Summary: {summary}\n"
                f"- Detected objects: {objects}\n"
                f"- License plates: {plates}\n"
                f"User question: {message}"
            )
            
            # Read and encode the image
            import base64
            import requests
            
            with open(analysis_request.file.path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
                
            gemma_api_url = os.getenv("GEMMA_API", "http://gemma:11434")
            model_name = os.getenv("MODEL_NAME", "gemma4:e2b")
            
            # Since the user requested stateless backend, we pass just the context + image + user message.
            llm_response = requests.post(
                f"{gemma_api_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": context,
                    "images": [image_data],
                    "stream": False,
                },
                timeout=120,
            )
            llm_response.raise_for_status()
            generated_text = llm_response.json().get("response", "").strip()
            
            return Response({"response": generated_text}, status=status.HTTP_200_OK)
            
        except ImageAnalysisRequest.DoesNotExist:
            return Response({"error": "Task not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            logger.exception("Error in chat endpoint for task %s: %s", pk, exc)
            return Response(
                {"error": "Internal server error", "details": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


def analysis_ui(request: Request) -> Response:
    """Display a simple upload page for frontend testing."""
    from django.shortcuts import render

    return render(request, "api/upload.html")
