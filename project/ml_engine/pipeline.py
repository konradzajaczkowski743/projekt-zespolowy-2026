"""
ML engine pipeline orchestration for the project.

This module defines ``run_full_analysis``, the primary function that
orchestrates the complete synchronous ML pipeline for a submitted analysis job.

Architecture notes:
    - The pipeline touches the database *and* the ML service layer: it updates 
      job status, calls the service classes in sequence, aggregates results,
      and persists the ``AnalysisResult`` record.
    - Image description from Gemma is passed to the geo-verification module
      to supplement visual location detection.
    - Distance estimation runs after object detection and uses its results.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from api.models import AnalysisResult, ImageAnalysisRequest
from ml_engine.services.alpr import ALPRAnalyzer
from ml_engine.services.distance_estimator import DistanceEstimator
from ml_engine.services.forensics import ImageForensicsAnalyzer
from ml_engine.services.geolocator import GeoVerificationAnalyzer
from ml_engine.services.image_description import ImageDescriptionAnalyzer
from ml_engine.services.object_det import ObjectDetector

logger: logging.Logger = logging.getLogger(__name__)


def run_full_analysis(task_id: str) -> dict[str, Any]:
    """
    Orchestrate the complete ML analysis pipeline for a single job.

    This function is dispatched by Celery via ``run_analysis_task``
    after the ``ImageAnalysisRequest`` record is created.

    Pipeline steps (executed sequentially to share model-load overhead):

    1. Transition job status to ``PROCESSING``.
    2. Run ``ImageForensicsAnalyzer.analyze()`` — EXIF + ELA + ML ensemble.
    3. Run ``ImageDescriptionAnalyzer.describe_image()`` — Gemma4 VLM description
       (only if image is authenticated as genuine).
    4. Run ``GeoVerificationAnalyzer.verify_location()`` — 3-layer visual
       geolocation (ViT + StreetCLIP + Gemma), with description as supplementary
       input.
    5. Run ``ObjectDetector.detect_objects()`` — RT-DETR object detection.
    6. Run ``DistanceEstimator.estimate_distance()`` — Depth Anything V2 depth
       map + object detection fusion.
    7. Run ``ALPRAnalyzer.recognize_plates()`` — licence plate recognition.
    8. Persist an ``AnalysisResult`` record with all outputs.
    9. Transition job status to ``COMPLETED``.

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
        image_paths: list[str] = [analysis_request.file.path]
        latitude: float | None = analysis_request.latitude
        longitude: float | None = analysis_request.longitude

        # Step 1 — Image forensics -------------------------------------------
        logger.debug("task_id=%s | Running ImageForensicsAnalyzer", task_id)
        forensics_analyzer = ImageForensicsAnalyzer()
        forensics_result: dict[str, Any] = forensics_analyzer.analyze(
            image_paths=image_paths
        )

        # Step 2 — Geo verification (ViT & StreetCLIP) -----------------------
        logger.debug("task_id=%s | Running GeoVerificationAnalyzer", task_id)
        geo_analyzer = GeoVerificationAnalyzer()
        geo_result: dict[str, Any] = geo_analyzer.verify_location(
            image_paths=image_paths,
            latitude=None,
            longitude=None,
        )

        # Step 3 — Image description & Arbitration (if authentic) ------------
        image_description_result: dict[str, Any] = {}
        is_authentic: bool = forensics_result.get("is_authentic", False)

        if is_authentic:
            logger.debug(
                "task_id=%s | Image is authentic, generating description and arbitrating with Gemma",
                task_id,
            )
            description_analyzer = ImageDescriptionAnalyzer()
            image_description_result = description_analyzer.describe_and_arbitrate(
                image_paths=image_paths,
                vit_label=geo_result.get("vit_predicted_region"),
                streetclip_results=geo_result.get("streetclip_top5", [])
            )
            
            arbitrated_loc = image_description_result.get("arbitrated_location")
            # Nie pozwól Gemmie nadpisać pewnego wyniku z wektorowej bazy obrazów!
            is_vector_db_hit = geo_result.get("prediction_source") == "vector_db"
            
            if is_vector_db_hit:
                # Vector DB jest najwyższym autorytetem — nadpisz arbitrację Gemmy
                image_description_result["arbitrated_location"] = geo_result["predicted_region"]
                logger.debug("task_id=%s | Vector DB hit — overriding Gemma arbitration to: %s", task_id, geo_result["predicted_region"])
            elif arbitrated_loc and str(arbitrated_loc).upper() != "UNKNOWN":
                geo_result["predicted_region"] = arbitrated_loc
                geo_result["prediction_source"] = "gemma_arbitration"
                geo_result["confidence_score"] = 0.90
                logger.debug("task_id=%s | Gemma arbitrated location: %s", task_id, arbitrated_loc)
            else:
                logger.debug("task_id=%s | Image description completed, no arbitration override", task_id)
        else:
            logger.debug(
                "task_id=%s | Image not authentic, skipping Gemma description",
                task_id,
            )

        # Calculate GPS distance based on the final predicted_region
        if latitude is not None and longitude is not None:
            if geo_result.get("predicted_region"):
                is_consistent, dist_km = geo_analyzer.calculate_distance(
                    geo_result["predicted_region"], latitude, longitude
                )
                geo_result["is_location_consistent"] = is_consistent
                geo_result["distance_km"] = dist_km
            else:
                geo_result["is_location_consistent"] = True
                geo_result["distance_km"] = None

        # Step 4 — Object detection ------------------------------------------
        logger.debug("task_id=%s | Running ObjectDetector (RT-DETR)", task_id)
        object_detector = ObjectDetector()
        objects_result: list[dict[str, Any]] = object_detector.detect_objects(
            image_paths=image_paths
        )

        # Step 5 — Distance estimation ---------------------------------------
        logger.debug("task_id=%s | Running DistanceEstimator", task_id)
        distance_estimator = DistanceEstimator()
        distance_result: dict[str, Any] = distance_estimator.estimate_distance(
            image_paths=image_paths,
            detected_objects=objects_result,
            scene_type=geo_result.get("predicted_region"),
            geo_verification=geo_result,
        )

        # Step 6 — ALPR -------------------------------------------------------
        logger.debug("task_id=%s | Running ALPRAnalyzer", task_id)
        alpr_analyzer = ALPRAnalyzer()
        alpr_result: list[dict[str, Any]] = alpr_analyzer.recognize_plates(
            image_paths=image_paths
        )

        # Step 7 — Cross-validation (Gemma vs RT-DETR) ------------------------
        if is_authentic and image_description_result:
            described_text = " ".join(image_description_result.get("objects_identified", [])).lower()
            detected_labels = set(obj["label"] for obj in objects_result)
            
            # Simple heuristic for "person"
            mentions_person = any(word in described_text for word in ["osoba", "ludzie", "człowiek", "przechodzień", "osoby", "tłum"])
            detects_person = "person" in detected_labels
            
            # Simple heuristic for "car"
            mentions_car = any(word in described_text for word in ["samochód", "auto", "pojazd", "samochody", "auta"])
            detects_car = any(label in detected_labels for label in ["car", "truck", "bus"])
            
            penalty = 0.0
            if mentions_person and not detects_person:
                penalty += 15.0
                logger.info("task_id=%s | Cross-val mismatch: Gemma mentions person, RT-DETR found none", task_id)
            if mentions_car and not detects_car:
                penalty += 15.0
                logger.info("task_id=%s | Cross-val mismatch: Gemma mentions car, RT-DETR found none", task_id)
                
            if penalty > 0:
                old_score = forensics_result.get("confidence_score", 100.0)
                new_score = max(0.0, old_score - penalty)
                forensics_result["confidence_score"] = new_score
                if new_score < 55.0:
                    forensics_result["is_authentic"] = False
                    forensics_result["manipulation_type"] = "inconsistent_objects"
                logger.warning("task_id=%s | Applied forensics penalty -%.1f due to object mismatch", task_id, penalty)

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
                "image_description": image_description_result,
                "distance_estimation": distance_result,
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
