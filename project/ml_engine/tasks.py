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
