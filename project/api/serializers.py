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
        help_text=(
            "Authenticity percentage in the range [0.0, 100.0]. "
            "0 means the image is classified as AI-generated, 100 means it is most likely real."
        ),
    )
    manipulation_type: serializers.CharField = serializers.CharField(
        allow_null=True,
        help_text="Detected manipulation category, or null if none detected.",
    )
    exif_anomalies: serializers.ListField = serializers.ListField(
        child=serializers.CharField(),
        help_text="List of human-readable EXIF anomaly descriptions.",
    )


class StreetCLIPPredictionSerializer(serializers.Serializer):
    """Serializer for a single StreetCLIP zero-shot prediction."""

    label: serializers.CharField = serializers.CharField(
        help_text="Candidate location label.",
    )
    score: serializers.FloatField = serializers.FloatField(
        help_text="Probability score for this location.",
    )


class GeoVerificationResultSerializer(serializers.Serializer):
    """
    Serializer for the geo-verification output section.

    Corresponds to the output produced by ``GeoVerificationAnalyzer``.
    Visual prediction is always independent of supplied GPS coordinates.
    """

    is_location_consistent: serializers.BooleanField = serializers.BooleanField(
        help_text="True when the visual scene is consistent with the given GPS co-ordinates.",
    )
    confidence_score: serializers.FloatField = serializers.FloatField(
        help_text="Geo consistency confidence in the range [0.0, 1.0].",
    )
    predicted_region: serializers.CharField = serializers.CharField(
        allow_null=True,
        help_text="Visually predicted geographic location (landmark, city, or region).",
    )
    vit_predicted_region: serializers.CharField = serializers.CharField(
        required=False,
        allow_null=True,
        help_text="ViT model's raw landmark prediction before arbitration.",
    )
    prediction_source: serializers.CharField = serializers.CharField(
        required=False,
        allow_null=True,
        help_text="Which layer made the prediction: vector_db, vit_landmarks, streetclip, or gemma_arbitration.",
    )
    distance_km: serializers.FloatField = serializers.FloatField(
        allow_null=True,
        help_text="Estimated distance (km) between predicted region and supplied GPS.",
    )
    streetclip_top5: StreetCLIPPredictionSerializer = StreetCLIPPredictionSerializer(
        many=True,
        required=False,
        help_text="Top-5 StreetCLIP zero-shot predictions with scores.",
    )
    description_location: serializers.CharField = serializers.CharField(
        required=False,
        allow_null=True,
        help_text="Location extracted from Gemma image description (supplementary signal).",
    )


class DetectedObjectSerializer(serializers.Serializer):
    """
    Serializer for a single detected object entry.

    Used as a child serializer within the ``objects_detected`` list.
    """

    label: serializers.CharField = serializers.CharField(
        help_text="Class label of the detected object.",
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


class ImageDescriptionSerializer(serializers.Serializer):
    """
    Serializer for image description output generated by Gemma LLM.

    Used when an image is authenticated as genuine.
    """

    description: serializers.CharField = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="Detailed description of image content generated by Gemma.",
    )
    confidence: serializers.FloatField = serializers.FloatField(
        required=False,
        help_text="Confidence score of the description (0.0 to 1.0).",
    )
    objects_identified: serializers.ListField = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        help_text="List of objects identified in the image.",
    )
    scenes: serializers.ListField = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        help_text="List of scenes or environments identified in the image.",
    )
    arbitrated_location: serializers.CharField = serializers.CharField(
        required=False,
        allow_null=True,
        help_text="Final arbitrated location — reflects the most confident prediction from all models.",
    )
    metadata: serializers.DictField = serializers.DictField(
        required=False,
        help_text="Metadata about the description generation (model used, processing time, etc).",
    )


class MainObjectSerializer(serializers.Serializer):
    """Serializer for the main object within distance estimation."""

    label: serializers.CharField = serializers.CharField(
        help_text="Label of the detected main object.",
    )
    confidence: serializers.FloatField = serializers.FloatField(
        help_text="Detection confidence of the main object.",
    )
    bounding_box: serializers.DictField = serializers.DictField(
        help_text="Bounding box of the main object.",
    )
    estimated_distance_m: serializers.FloatField = serializers.FloatField(
        allow_null=True,
        help_text="Estimated distance to the main object in metres.",
    )
    depth_value: serializers.FloatField = serializers.FloatField(
        help_text="Raw relative depth value from the depth map (0=close, 1=far).",
    )


class DistanceEstimationSerializer(serializers.Serializer):
    """
    Serializer for distance estimation output.

    When ``has_main_object`` is False, the image depicts a scene without
    a distinct main object (e.g. beach, landscape) and ``main_object``
    will be null.
    """

    has_main_object: serializers.BooleanField = serializers.BooleanField(
        help_text="True if a distinct main object was found in the image.",
    )
    main_object = MainObjectSerializer(
        required=False,
        allow_null=True,
        help_text="Details about the main object and its estimated distance.",
    )
    reason: serializers.CharField = serializers.CharField(
        required=False,
        allow_null=True,
        help_text="Explanation when no main object is found (e.g. 'landscape_scene_no_main_object').",
    )
    depth_map_stats: serializers.DictField = serializers.DictField(
        required=False,
        allow_null=True,
        help_text="Statistics of the depth map (min, max, mean depth values).",
    )
    model_used: serializers.CharField = serializers.CharField(
        required=False,
        help_text="Depth estimation model version used.",
    )


# Input serializer


class AnalysisRequestSerializer(serializers.Serializer):
    """
    Input serializer for a new image analysis request.

    Validates the multipart/form-data payload submitted by the client.

    Fields:
        image: Image file to analyse (JPEG / PNG).
        latitude: Optional GPS latitude in decimal degrees (-90 to 90).
        longitude: Optional GPS longitude in decimal degrees (-180 to 180).
    """

    image: serializers.ImageField = serializers.ImageField(
        required=False,
        allow_null=True,
        help_text="Image file to analyse.",
    )
    image_url: serializers.URLField = serializers.URLField(
        required=False,
        allow_null=True,
        allow_blank=False,
        help_text="Image URL to download and analyse.",
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

    def validate_image(self, value: serializers.ImageField) -> serializers.ImageField:
        """
        Validate that the uploaded file has an acceptable MIME type.

        Args:
            value: Uploaded ``InMemoryUploadedFile`` object.

        Returns:
            The validated image file.

        Raises:
            serializers.ValidationError: If the file has an unsupported type.
        """
        allowed_content_types: set[str] = {"image/jpeg", "image/png", "image/webp"}
        if value.content_type not in allowed_content_types:
            raise serializers.ValidationError(
                f"Unsupported file type '{value.content_type}'. "
                f"Allowed types: {', '.join(sorted(allowed_content_types))}."
            )
        return value

    def validate_image_url(self, value: str) -> str:
        if not value.lower().startswith(("http://", "https://")):
            raise serializers.ValidationError(
                "URL must start with http:// or https://"
            )
        return value

    def validate(self, data: dict[str, Any]) -> dict[str, Any]:
        image = data.get("image")
        image_url = data.get("image_url")

        if image and image_url:
            raise serializers.ValidationError(
                "Provide either an image file or an image URL, not both."
            )

        if not image and not image_url:
            raise serializers.ValidationError(
                "Upload an image file or provide an image URL to analyse."
            )

        return data


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
    image_description = ImageDescriptionSerializer(
        required=False,
        allow_null=True,
        help_text="Detailed image description generated by Gemma LLM (only if image is authentic).",
    )
    geo_verification = GeoVerificationResultSerializer(
        required=False,
        allow_null=True,
        help_text="Geo-verification analysis output (3-layer visual recognition).",
    )
    distance_estimation = DistanceEstimationSerializer(
        required=False,
        allow_null=True,
        help_text="Distance estimation to main object (Depth Anything V2 + RT-DETR).",
    )

    class Meta:
        """Metadata for the AnalysisResultSerializer."""

        model = AnalysisResult
        fields = [
            "forensics",
            "image_description",
            "geo_verification",
            "distance_estimation",
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
