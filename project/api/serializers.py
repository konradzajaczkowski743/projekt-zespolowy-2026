"""
DRF serializers for the project.
"""
from __future__ import annotations

from rest_framework import serializers

from api.models import AnalysisResult, ImageAnalysisRequest


# Serializers for specific ML output fields


class ForensicsResultSerializer(serializers.Serializer):
    """
    Serializer for the forensics analysis output section.

    Corresponds to the output produced by ``ImageForensicsAnalyzer``.
    All fields are read-only; this serializer is used only for responses.
    """

    is_authentic: serializers.BooleanField = serializers.BooleanField(
        help_text="True when the image passes all forensic checks.",
    )
    confidence_score: serializers.FloatField = serializers.FloatField(
        help_text="Authenticity confidence in the range [0.0, 1.0].",
    )
    manipulation_type: serializers.CharField = serializers.CharField(
        allow_null=True,
        help_text="Detected manipulation category, or null if none detected.",
    )
    exif_anomalies: serializers.ListField = serializers.ListField(
        child=serializers.CharField(),
        help_text="List of human-readable EXIF anomaly descriptions.",
    )
    noise_analysis: serializers.DictField = serializers.DictField(
        help_text="Pixel-level noise analysis metrics.",
    )


class GeoVerificationResultSerializer(serializers.Serializer):
    """
    Serializer for the geo-verification output section.

    Corresponds to the output produced by ``GeoVerificationAnalyzer``.
    """

    is_location_consistent: serializers.BooleanField = serializers.BooleanField(
        help_text="True when the visual scene is consistent with the given GPS co-ordinates.",
    )
    confidence_score: serializers.FloatField = serializers.FloatField(
        help_text="Geo consistency confidence in the range [0.0, 1.0].",
    )
    predicted_region: serializers.CharField = serializers.CharField(
        allow_null=True,
        help_text="Statistical prediction of the geographic region visible in the image.",
    )
    distance_km: serializers.FloatField = serializers.FloatField(
        allow_null=True,
        help_text="Estimated distance (km) between predicted region and supplied GPS.",
    )


class DetectedObjectSerializer(serializers.Serializer):
    """
    Serializer for a single detected object entry.

    Used as a child serializer within the ``objects_detected`` list.
    """

    label: serializers.CharField = serializers.CharField(
        help_text="lass label of the detected object.",
    )
    confidence: serializers.FloatField = serializers.FloatField(
        help_text="Detection confidence in the range [0.0, 1.0].",
    )
    bounding_box: serializers.DictField = serializers.DictField(
        help_text=(
            "Bounding box dict with keys 'x_min', 'y_min', 'x_max', 'y_max' "
            "as pixel co-ordinates."
        ),
    )


class ALPRResultSerializer(serializers.Serializer):
    """
    Serializer for a single ALPR (Automatic Licence Plate Recognition) entry.

    Used as a child serializer within the ``alpr`` list.
    """

    plate_text: serializers.CharField = serializers.CharField(
        help_text="Transcribed licence plate text.",
    )
    confidence: serializers.FloatField = serializers.FloatField(
        help_text="Recognition confidence in the range [0.0, 1.0].",
    )
    country_code: serializers.CharField = serializers.CharField(
        allow_null=True,
        help_text="ISO 3166-1 alpha-2 country code inferred from plate format.",
    )
    bounding_box: serializers.DictField = serializers.DictField(
        help_text="Bounding box of the plate region (same format as DetectedObjectSerializer).",
    )


# Input serializer


class AnalysisRequestSerializer(serializers.Serializer):
    """
    Input serializer for a new image analysis request.

    Validates the multipart/form-data payload submitted by the client.
    At least one image file must be provided.

    Fields:
        images: One or more image files (JPEG / PNG).
        latitude: Optional GPS latitude in decimal degrees (-90 to 90).
        longitude: Optional GPS longitude in decimal degrees (-180 to 180).
    """

    images: serializers.ListField = serializers.ListField(
        child=serializers.ImageField(),
        allow_empty=False,
        min_length=1,
        max_length=20,
        help_text="List of image files to analyse.",
    )
    latitude: serializers.FloatField = serializers.FloatField(
        required=False,
        allow_null=True,
        min_value=-90.0,
        max_value=90.0,
        help_text="GPS latitude in decimal degrees (optional).",
    )
    longitude: serializers.FloatField = serializers.FloatField(
        required=False,
        allow_null=True,
        min_value=-180.0,
        max_value=180.0,
        help_text="GPS longitude in decimal degrees (optional).",
    )

    def validate_images(self, value: list) -> list:
        """
        Validate that every uploaded file has an acceptable MIME type.

        Args:
            value: List of uploaded ``InMemoryUploadedFile`` objects.

        Returns:
            The validated list of image files.

        Raises:
            serializers.ValidationError: If any file has an unsupported type.
        """
        allowed_content_types: set[str] = {"image/jpeg", "image/png", "image/webp"}
        for image_file in value:
            if image_file.content_type not in allowed_content_types:
                raise serializers.ValidationError(
                    f"Unsupported file type '{image_file.content_type}'. "
                    f"Allowed types: {', '.join(sorted(allowed_content_types))}."
                )
        return value


# Output serializers


class AnalysisResultSerializer(serializers.ModelSerializer):
    """
    Full serializer for a completed ``AnalysisResult`` database record.

    Nested ML output sub-serializers are used to reflect the agreed JSON
    contract.  This serializer is read-only (used only in GET responses).
    """

    forensics = ForensicsResultSerializer(
        help_text="Image forensics analysis output.",
    )
    geo_verification = GeoVerificationResultSerializer(
        help_text="Geo-verification analysis output.",
    )
    objects_detected = DetectedObjectSerializer(
        many=True,
        help_text="List of objects detected in the image.",
    )
    alpr = ALPRResultSerializer(
        many=True,
        help_text="List of recognised licence plates.",
    )

    class Meta:
        """Metadata for the AnalysisResultSerializer."""

        model = AnalysisResult
        fields = [
            "processing_time_ms",
            "forensics",
            "geo_verification",
            "objects_detected",
            "alpr",
            "error_message",
            "created_at",
        ]


class AnalysisStatusSerializer(serializers.ModelSerializer):
    """
    Lightweight serializer used for both 202-Accepted and GET status responses.

    Returns a minimal payload so that clients can begin polling without
    receiving the full (potentially large) result payload.
    """

    task_id: serializers.UUIDField = serializers.UUIDField(
        source="id",
        read_only=True,
        help_text="UUID identifying this analysis job — use for GET /status/<task_id>/.",
    )
    result = AnalysisResultSerializer(
        read_only=True,
        required=False,
        allow_null=True,
        help_text="Full ML result payload; null until the job is COMPLETED.",
    )

    class Meta:
        """Metadata for the AnalysisStatusSerializer."""

        model = ImageAnalysisRequest
        fields = [
            "task_id",
            "status",
            "created_at",
            "updated_at",
            "result",
        ]
