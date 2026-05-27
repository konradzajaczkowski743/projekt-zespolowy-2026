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

    logger.info("Celery task started: process_image_query_task query_id=%s", query_id)

    try:
        query = ImageQuery.objects.get(id=query_id)
        
        # Get the analysis result and image path
        result = query.result
        image_path = result.request.file.path
        
        # Initialize the image description analyzer
        analyzer = ImageDescriptionAnalyzer()
        
        # Get analysis context for better responses
        analysis_context = {
            "forensics": result.forensics,
            "geo_verification": result.geo_verification,
            "objects_detected": result.objects_detected,
            "distance_estimation": result.distance_estimation,
            "image_description": result.image_description,
        }
        
        # Call Gemma LLM with the query and analysis context
        prompt = (
            "Jesteś asystentem AI pomagającym użytkownikom zrozumieć wyniki analizy zdjęcia.\n"
            "Użytkownik zadał następujące pytanie dotyczące analizy obrazu:\n"
            f"Pytanie: {query.user_query}\n\n"
            "Masz dostęp do wyników analizy:\n"
            f"Kontekst analizy: {analysis_context}\n\n"
            "Odpowiedz na pytanie użytkownika w jasny, zrozumiały sposób na podstawie dostępnych danych.\n"
            "Jeśli nie masz informacji, aby odpowiedzieć na pytanie, wyjaśnij, dlaczego."
        )
        
        response = analyzer.query_image_with_context(image_path, prompt)
        
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
