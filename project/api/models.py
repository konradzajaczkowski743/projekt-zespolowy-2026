"""
Database models for the project API application.

Two primary models are defined here:

* ``ImageAnalysisRequest`` - represents a single analysis job and its associated
  image file. It stores job lifecycle metadata as well as technical details
  like dimensions and geolocation.

* ``AnalysisResult`` - stores the JSON payload produced by ML workers once
  processing is complete.
"""

import uuid

from django.db import models

class ImageAnalysisRequest(models.Model):
    """
    Represents a single image-analysis request submitted through the REST API.

    This model tracks the overall lifecycle of the asynchronous processing job
    and stores the physical file reference along with metadata extracted during
    the upload phase (e.g., EXIF data, dimensions).

    Attributes:
        id: UUID primary key used as the public 'task_id' in API interactions.
        status: Current state of the job (PENDING, PROCESSING, COMPLETED, FAILED).
        file: ImageField managing the physical file storage and path generation.
        original_filename: The name of the file as submitted by the client.
        file_size: Size of the image file in bytes.
        width: Image width in pixels (auto-populated by ImageField).
        height: Image height in pixels (auto-populated by ImageField).
        latitude: GPS latitude extracted from EXIF or provided via API.
        longitude: GPS longitude extracted from EXIF or provided via API.
        exif_data: Dictionary containing raw EXIF metadata tags.
        started_at: UTC timestamp when an ML worker first picked up the task.
        completed_at: UTC timestamp when the image finished processing.
        created_at: UTC timestamp when the request was first persisted.
        updated_at: UTC timestamp of the last modification to the request state.
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
        help_text="Unique identifier for this analysis job.",
    )
    status: models.CharField = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
        help_text="Current lifecycle state of the analysis job.",
    )
    file: models.ImageField = models.ImageField(
        upload_to="analysis_images/%Y/%m/%d/",
        width_field="width",
        height_field="height",
        help_text="The physical image file stored on the local filesystem.",
    )
    original_filename: models.CharField = models.CharField(
        max_length=255,
        blank=True,
        help_text="The original filename for reference.",
    )
    file_size: models.PositiveIntegerField = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="File size in bytes.",
    )
    width: models.IntegerField = models.IntegerField(
        null=True,
        blank=True,
        help_text="Automated pixel width.",
    )
    height: models.IntegerField = models.IntegerField(
        null=True,
        blank=True,
        help_text="Automated pixel height.",
    )
    latitude: models.FloatField = models.FloatField(
        null=True,
        blank=True,
        help_text="Geospatial latitude coordinate.",
    )
    longitude: models.FloatField = models.FloatField(
        null=True,
        blank=True,
        help_text="Geospatial longitude coordinate.",
    )
    exif_data: models.JSONField = models.JSONField(
        default=dict,
        blank=True,
        help_text="Extracted EXIF metadata in JSON format.",
    )
    started_at: models.DateTimeField = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the ML pipeline began processing this request.",
    )
    completed_at: models.DateTimeField = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the ML pipeline finished or encountered a fatal error.",
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

    Has a one-to-one relationship with ``ImageAnalysisRequest``. Results are typically
    written once the ML pipeline has finished all inference tasks for the file.

    Attributes:
        request: The specific ImageAnalysisRequest this result describes.
        is_deepfake: Boolean flag for high-level deepfake classification.
        deepfake_score: Confidence probability of the deepfake detection (0.0 - 1.0).
        forensics: JSON output detailing image manipulation analysis.
        geo_verification: JSON output comparing EXIF data with visual cues.
        objects_detected: JSON list of identified objects and their bounding boxes.
        alpr: JSON output from Automated License Plate Recognition.
        processing_time_ms: Total time in milliseconds spent on ML inference.
        error_message: Detailed error description if the analysis failed.
        created_at: UTC timestamp when the result record was created.
    """

    request: models.OneToOneField = models.OneToOneField(
        ImageAnalysisRequest,
        on_delete=models.CASCADE,
        related_name="result",
        help_text="The specific analysis request this result belongs to.",
    )
    is_deepfake: models.BooleanField = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Indicates if the image was flagged as a potential deepfake.",
    )
    deepfake_score: models.FloatField = models.FloatField(
        null=True,
        blank=True,
        help_text="The probability score from the deepfake detection model.",
    )
    forensics: models.JSONField = models.JSONField(
        default=dict,
        blank=True,
        help_text="Structured data from the forensics analysis module.",
    )
    geo_verification: models.JSONField = models.JSONField(
        default=dict,
        blank=True,
        help_text="Cross-reference results between metadata and visual landmarks.",
    )
    objects_detected: models.JSONField = models.JSONField(
        default=list,
        blank=True,
        help_text="A list of detected objects with confidence and coordinates.",
    )
    alpr: models.JSONField = models.JSONField(
        default=list,
        blank=True,
        help_text="Detected vehicle license plate information.",
    )
    processing_time_ms: models.IntegerField = models.IntegerField(
        default=0,
        help_text="Execution time of the ML pipeline in milliseconds.",
    )
    error_message: models.TextField = models.TextField(
        blank=True,
        default="",
        help_text="Populated with traceback or error info if analysis fails.",
    )
    created_at: models.DateTimeField = models.DateTimeField(auto_now_add=True)

    class Meta:
        """Django model metadata."""

        verbose_name = "Analysis Result"
        verbose_name_plural = "Analysis Results"

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"AnalysisResult(request={self.request_id}, deepfake={self.is_deepfake})"