"""
Database models for the the project API application.

Two primary models are defined here:

* ``ImageAnalysisRequest`` — represents a single analysis job submitted by a
  client.  It owns the uploaded images and records lifecycle metadata.

* ``AnalysisResult`` — stores the JSON payload produced by ML workers once
  processing is complete.

"""
import uuid

from django.db import models


class ImageAnalysisRequest(models.Model):
    """
    Represents a single image-analysis request submitted through the REST API.

    Attributes:
        id: UUID primary key exposed as the ``task_id`` in API responses.
        status: Lifecycle state of the analysis job.
        images: JSON list of relative paths to uploaded image files.
        latitude: Optional GPS latitude supplied by the client.
        longitude: Optional GPS longitude supplied by the client.
        created_at: UTC timestamp when the request was created.
        updated_at: UTC timestamp of the last status change.
    """

    class Status(models.TextChoices):
        """Enumeration of allowed lifecycle states for an analysis request."""

        PENDING = "PENDING", "Pending"
        PROCESSING = "PROCESSING", "Processing"
        COMPLETED = "COMPLETED", "Completed"
        FAILED = "FAILED", "Failed"

    id: models.UUIDField = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Unique identifier for this analysis job (exposed as task_id).",
    )
    status: models.CharField = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
        help_text="Current lifecycle state of the analysis job.",
    )
    images: models.JSONField = models.JSONField(
        default=list,
        help_text="List of relative media paths for the uploaded images.",
    )
    latitude: models.FloatField = models.FloatField(
        null=True,
        blank=True,
        help_text="Optional GPS latitude provided by the client.",
    )
    longitude: models.FloatField = models.FloatField(
        null=True,
        blank=True,
        help_text="Optional GPS longitude provided by the client.",
    )
    created_at: models.DateTimeField = models.DateTimeField(auto_now_add=True)
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True)

    class Meta:
        """Django model metadata."""

        ordering = ["-created_at"]
        verbose_name = "Image Analysis Request"
        verbose_name_plural = "Image Analysis Requests"

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"AnalysisRequest({self.id}, status={self.status})"


class AnalysisResult(models.Model):
    """
    Stores the structured ML output for a completed analysis request.

    Has a one-to-one relationship with ``ImageAnalysisRequest``.  The result
    is written synchronously once all ML services have finished.

    Attributes:
        request: The parent ``ImageAnalysisRequest`` this result belongs to.
        processing_time_ms: Wall-clock time (ms) taken by the ML pipeline.
        forensics: JSON output from the image forensics analyser.
        geo_verification: JSON output from the geo-verification analyser.
        objects_detected: JSON output from the object detector.
        alpr: JSON output from the ALPR analyser.
        error_message: Human-readable error description when status is FAILED.
        created_at: UTC timestamp when the result record was created.
    """

    request: models.OneToOneField = models.OneToOneField(
        ImageAnalysisRequest,
        on_delete=models.CASCADE,
        related_name="result",
        help_text="The analysis job this result belongs to.",
    )
    processing_time_ms: models.IntegerField = models.IntegerField(
        default=0,
        help_text="Total wall-clock time in milliseconds for the ML pipeline.",
    )
    forensics: models.JSONField = models.JSONField(
        default=dict,
        help_text="Structured output from the ImageForensicsAnalyzer service.",
    )
    geo_verification: models.JSONField = models.JSONField(
        default=dict,
        help_text="Structured output from the GeoVerificationAnalyzer service.",
    )
    objects_detected: models.JSONField = models.JSONField(
        default=list,
        help_text="Structured output from the ObjectDetector service.",
    )
    alpr: models.JSONField = models.JSONField(
        default=list,
        help_text="Structured output from the ALPRAnalyzer service.",
    )
    error_message: models.TextField = models.TextField(
        blank=True,
        default="",
        help_text="Error description populated when the job status is FAILED.",
    )
    created_at: models.DateTimeField = models.DateTimeField(auto_now_add=True)

    class Meta:
        """Django model metadata."""

        verbose_name = "Analysis Result"
        verbose_name_plural = "Analysis Results"

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"AnalysisResult(request={self.request_id})"
