"""
ML engine pipeline orchestration for the project.

This module defines ``run_full_analysis``, the primary function that
orchestrates the complete synchronous ML pipeline for a submitted analysis job.

Architecture notes:
    - The pipeline touches the database *and* the ML service layer: it updates 
      job status, calls the service classes in sequence, aggregates results,
      and persists the ``AnalysisResult`` record.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from api.models import AnalysisResult, ImageAnalysisRequest
from ml_engine.services.alpr import ALPRAnalyzer
from ml_engine.services.forensics import ImageForensicsAnalyzer
from ml_engine.services.geolocator import GeoVerificationAnalyzer
from ml_engine.services.object_det import ObjectDetector

logger: logging.Logger = logging.getLogger(__name__)


def run_full_analysis(task_id: str) -> dict[str, Any]:
    """
    Orchestrate the complete ML analysis pipeline for a single job.

    This function is dispatched by ``AnalysisViewSet.create`` synchronously
    immediately after the ``ImageAnalysisRequest`` record is created.

    Pipeline steps (executed sequentially to share model-load overhead):

    1. Transition job status to ``PROCESSING``.
    2. Run ``ImageForensicsAnalyzer.analyze_exif()`` and ``detect_tampering()``.
    3. Run ``GeoVerificationAnalyzer.verify_location()``.
    4. Run ``ObjectDetector.detect_objects()``.
    5. Run ``ALPRAnalyzer.recognize_plates()``.
    6. Persist an ``AnalysisResult`` record with all outputs.
    7. Transition job status to ``COMPLETED``.

    On any unhandled exception the job is transitioned to ``FAILED``, the
    error message is recorded, and the exception is re-raised.

    Args:
        task_id: String UUID matching an ``ImageAnalysisRequest.id`` record.

    Returns:
        Dictionary with the key ``"task_id"`` identifying the completed run.

    Raises:
        ImageAnalysisRequest.DoesNotExist: If no matching record is found.
        Exception: Any ML service exception after recording ``FAILED`` status.
    """
    logger.info("run_full_analysis started: task_id=%s", task_id)

    # -- Fetch the DB record -------------------------------------------------
    try:
        analysis_request: ImageAnalysisRequest = ImageAnalysisRequest.objects.get(
            pk=task_id
        )
    except ImageAnalysisRequest.DoesNotExist:
        logger.error(
            "run_full_analysis: AnalysisRequest not found: task_id=%s", task_id
        )
        raise

    # -- Transition to PROCESSING --------------------------------------------
    analysis_request.status = ImageAnalysisRequest.Status.PROCESSING
    analysis_request.save(update_fields=["status", "updated_at"])

    pipeline_start: float = time.monotonic()

    try:
        image_paths: list[str] = analysis_request.images
        latitude: float | None = analysis_request.latitude
        longitude: float | None = analysis_request.longitude

        # Step 1 — Image forensics -------------------------------------------
        logger.debug("task_id=%s | Running ImageForensicsAnalyzer", task_id)
        forensics_analyzer = ImageForensicsAnalyzer()
        forensics_result: dict[str, Any] = forensics_analyzer.analyze(
            image_paths=image_paths
        )

        # Step 2 — Geo verification ------------------------------------------
        logger.debug("task_id=%s | Running GeoVerificationAnalyzer", task_id)
        geo_analyzer = GeoVerificationAnalyzer()
        geo_result: dict[str, Any] = geo_analyzer.verify_location(
            image_paths=image_paths,
            latitude=latitude,
            longitude=longitude,
        )

        # Step 3 — Object detection ------------------------------------------
        logger.debug("task_id=%s | Running ObjectDetector", task_id)
        object_detector = ObjectDetector()
        objects_result: list[dict[str, Any]] = object_detector.detect_objects(
            image_paths=image_paths
        )

        # Step 4 — ALPR -------------------------------------------------------
        logger.debug("task_id=%s | Running ALPRAnalyzer", task_id)
        alpr_analyzer = ALPRAnalyzer()
        alpr_result: list[dict[str, Any]] = alpr_analyzer.recognize_plates(
            image_paths=image_paths
        )

        processing_time_ms: int = int(
            (time.monotonic() - pipeline_start) * 1000
        )

        # -- Persist results -------------------------------------------------
        AnalysisResult.objects.update_or_create(
            request=analysis_request,
            defaults={
                "processing_time_ms": processing_time_ms,
                "forensics": forensics_result,
                "geo_verification": geo_result,
                "objects_detected": objects_result,
                "alpr": alpr_result,
                "error_message": "",
            },
        )

        # -- Transition to COMPLETED -----------------------------------------
        analysis_request.status = ImageAnalysisRequest.Status.COMPLETED
        analysis_request.save(update_fields=["status", "updated_at"])

        logger.info(
            "run_full_analysis completed: task_id=%s processing_time_ms=%d",
            task_id,
            processing_time_ms,
        )

    except Exception as exc:
        # -- Record failure and re-raise -------------------------------------
        logger.exception(
            "run_full_analysis failed: task_id=%s error=%s", task_id, exc
        )
        AnalysisResult.objects.update_or_create(
            request=analysis_request,
            defaults={
                "processing_time_ms": int(
                    (time.monotonic() - pipeline_start) * 1000
                ),
                "error_message": str(exc),
            },
        )
        analysis_request.status = ImageAnalysisRequest.Status.FAILED
        analysis_request.save(update_fields=["status", "updated_at"])
        raise exc

    return {"task_id": task_id}
