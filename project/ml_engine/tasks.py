"""
Celery task wrappers for ML pipeline operations.

Provides asynchronous task entry-points that delegate to the synchronous
pipeline functions defined in ``ml_engine.pipeline``.
"""
from __future__ import annotations

import logging

from celery import shared_task

logger: logging.Logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    name="ml_engine.run_analysis",
    max_retries=1,
    default_retry_delay=30,
    acks_late=True,
)
def run_analysis_task(self, task_id: str) -> dict:
    """
    Execute the full ML analysis pipeline as a background Celery task.

    This task is dispatched by ``AnalysisViewSet.create`` immediately after
    the ``ImageAnalysisRequest`` record is persisted, allowing the HTTP
    response to return a 202 Accepted right away.

    Args:
        task_id: String UUID matching an ``ImageAnalysisRequest.id`` record.

    Returns:
        Dictionary with the key ``"task_id"`` identifying the completed run.
    """
    from ml_engine.pipeline import run_full_analysis

    logger.info("Celery task started: run_analysis_task task_id=%s", task_id)

    try:
        result = run_full_analysis(task_id)
        logger.info("Celery task completed: task_id=%s", task_id)
        return result
    except Exception as exc:
        logger.exception(
            "Celery task failed: task_id=%s error=%s", task_id, exc
        )
        raise self.retry(exc=exc)


@shared_task(
    bind=True,
    name="ml_engine.process_image_query",
    max_retries=1,
    default_retry_delay=30,
    acks_late=True,
)
def process_image_query_task(self, query_id: str) -> dict:
    """
    Process a user query about an analyzed image using Gemma LLM.

    This task is dispatched when a user submits a question about a previously
    analyzed image. The query is processed by Gemma LLM and stored in the
    ImageQuery record.

    Args:
        query_id: String UUID matching an ``ImageQuery.id`` record.

    Returns:
        Dictionary with the key ``"query_id"`` and the generated response.
    """
    from api.models import ImageQuery
    from ml_engine.services.image_description import ImageDescriptionAnalyzer
    from ml_engine.services.alpr import ALPRAnalyzer
    from ml_engine.services.object_det import ObjectDetector

    logger.info("Celery task started: process_image_query_task query_id=%s", query_id)

    try:
        query = ImageQuery.objects.get(id=query_id)

        # Get the analysis result and the image path
        result = query.result
        image_path = result.request.file.path if result and result.request and result.request.file else None

        # Initialize analyzers
        analyzer = ImageDescriptionAnalyzer()
        alpr_analyzer = ALPRAnalyzer()
        object_detector = ObjectDetector()

        # Use ONLY the previous analysis results by default
        analysis_data = {
            "description": result.image_description.get("description", "") if result.image_description else "",
            "objects_identified": result.image_description.get("objects_identified", []) if result.image_description else [],
            "scenes": result.image_description.get("scenes", []) if result.image_description else [],
            "arbitrated_location": result.image_description.get("arbitrated_location") if result.image_description else None,
            "objects_detected": result.objects_detected or [],
            "alpr": result.alpr or [],
        }

        logger.debug(
            "Processing query_id=%s with analysis context (no image reprocessing by default)",
            query_id,
        )

        # If the user explicitly asks about licence plates and ALPR is missing,
        # run a targeted ALPR pass and persist the results before answering.
        lower_q = (query.user_query or "").lower()
        try:
            if image_path and any(k in lower_q for k in ("tablic", "tablica", "rejestr", "nr ", "nr.", "numer")) and not analysis_data.get("alpr"):
                logger.debug("Query requests ALPR but previous result empty — running ALPRAnalyzer")
                alpr_result = alpr_analyzer.recognize_plates([image_path])
                # persist to DB
                result.alpr = alpr_result
                result.save(update_fields=["alpr"])  # AnalysisResult has alpr field
                analysis_data["alpr"] = alpr_result

            # If the user asks about counts/number of objects and objects_detected is empty,
            # run object detection targeted pass and persist.
            if image_path and ("ile" in lower_q or "liczb" in lower_q or "ile jest" in lower_q or "how many" in lower_q) and not analysis_data.get("objects_detected"):
                logger.debug("Query requests object counts but previous objects_detected empty — running ObjectDetector")
                objects_result = object_detector.detect_objects([image_path])
                result.objects_detected = objects_result
                result.save(update_fields=["objects_detected"])
                analysis_data["objects_detected"] = objects_result
        except Exception as e:
            logger.exception("Error during targeted re-analysis for query_id=%s: %s", query_id, e)

        def response_indicates_unknown(text: str) -> bool:
            if not text:
                return True
            text_lower = text.lower()
            return any(
                phrase in text_lower
                for phrase in (
                    "nie jest możliwe",
                    "nie można określić",
                    "na podstawie poprzedniej analizy",
                    "nie mogę określić",
                    "nie udało się określić",
                    "brak informacji",
                )
            )

        # Call Gemma LLM with the (possibly updated) analysis context
        response = analyzer.query_with_analysis_context(
            analysis_result=analysis_data,
            user_prompt=query.user_query,
        )

        # If the analysis-only response is insufficient, ask directly from the image.
        if image_path and response_indicates_unknown(response):
            logger.info(
                "Query_id=%s: context answer insufficient, retrying with image fallback",
                query_id,
            )
            fallback_prompt = (
                "Masz tylko to zdjęcie i pytanie użytkownika. "
                "Pomiń poprzednią analizę i skup się wyłącznie na obrazie oraz treści pytania.\n\n"
                f"Pytanie użytkownika: {query.user_query}\n\n"
                "Przeprowadź nową analizę obrazu i udziel precyzyjnej odpowiedzi wyłącznie na jego podstawie. "
                "Jeżeli obraz zawiera odpowiedź na pytanie, podaj ją bez domysłów.")
            try:
                response = analyzer.query_image_with_context(image_path, fallback_prompt)
            except Exception as e:
                logger.exception(
                    "Image fallback query failed for query_id=%s: %s",
                    query_id,
                    e,
                )

        # Store the response
        query.ai_response = response
        query.status = ImageQuery.Status.COMPLETED
        query.save(update_fields=["ai_response", "status", "updated_at"])

        logger.info("Celery task completed: query_id=%s", query_id)
        return {"query_id": str(query_id), "response": response}
        
    except ImageQuery.DoesNotExist:
        logger.error("ImageQuery not found: query_id=%s", query_id)
        return {"error": f"Query {query_id} not found"}
    except Exception as exc:
        logger.exception("Celery task failed: query_id=%s error=%s", query_id, exc)
        # Mark the query as failed
        try:
            query = ImageQuery.objects.get(id=query_id)
            query.status = ImageQuery.Status.FAILED
            query.error_message = str(exc)
            query.save(update_fields=["status", "error_message", "updated_at"])
        except:
            pass
        raise self.retry(exc=exc)
