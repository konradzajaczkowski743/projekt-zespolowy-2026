"""
Geo-verification analysis service stub for the project.

This module provides ``GeoVerificationAnalyzer``, a clean interface (mock) for
checking whether the visual content of an image is geographically consistent
with the GPS co-ordinates supplied by the submitting client.

Expected real-model integrations (to be added later):
    - Fully local scene-classification CNN (e.g., a fine-tuned ResNet).
    - Custom local geolocation regression model (IM2GPS3k-style).
    - Haversine distance computation between predicted region and supplied GPS.
    - No external routing APIs (like Google Maps) are permitted.
"""
from __future__ import annotations

import logging
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


class GeoVerificationAnalyzer:
    """
    Verifies whether image content is consistent with the provided GPS location.

    All public methods return data structures that match the JSON contract
    defined by ``GeoVerificationResultSerializer`` in ``api/serializers.py``.

    Attributes:
        model_version: Semantic version string of the underlying ML model.
    """

    model_version: str = "0.1.0-mock"

    def __init__(self) -> None:
        """Initialise the analyser and (eventually) load the ML model."""
        logger.debug(
            "GeoVerificationAnalyzer initialised (version=%s)", self.model_version
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def verify_location(
        self,
        image_paths: list[str],
        latitude: float | None,
        longitude: float | None,
    ) -> dict[str, Any]:
        """
        Assess whether the visual scene matches the claimed GPS co-ordinates.

        This method is the primary entry point called by the ML pipeline.  It
        delegates to ``predict_region`` to obtain the visual location estimate
        and then computes the geo-consistency score.

        Args:
            image_paths: List of relative media-root paths to the uploaded
                image files.
            latitude: GPS latitude in decimal degrees, or ``None`` if not
                supplied by the submitting client.
            longitude: GPS longitude in decimal degrees, or ``None`` if not
                supplied by the submitting client.

        Returns:
            Dictionary conforming to ``GeoVerificationResultSerializer``::

                {
                    "is_location_consistent": bool,
                    "confidence_score": float,     # [0.0, 1.0]
                    "predicted_region": str | None,
                    "distance_km": float | None
                }
        """
        logger.info(
            "GeoVerificationAnalyzer.verify_location: images=%d lat=%s lon=%s",
            len(image_paths),
            latitude,
            longitude,
        )

        predicted_region: str | None = self.predict_region(image_paths)

        # When no GPS was supplied, only the visual prediction is returned.
        is_consistent: bool = True
        distance_km: float | None = None

        if latitude is not None and longitude is not None and predicted_region:
            # TODO: Replace with real Haversine distance calculation between
            #       the centroid of ``predicted_region`` and the supplied GPS.
            distance_km = 12.5  # Mock: 12.5 km offset
            is_consistent = distance_km < 50.0

        result: dict[str, Any] = {
            "is_location_consistent": is_consistent,
            "confidence_score": 0.85,
            "predicted_region": predicted_region,
            "distance_km": distance_km,
        }
        logger.debug("GeoVerificationAnalyzer.verify_location result: %s", result)
        return result

    def predict_region(self, image_paths: list[str]) -> str | None:
        """
        Predict the geographic region depicted in the supplied images.

        In production this will run images through a fully local geolocation-regression
        CNN (e.g. a fine-tuned EfficientNet trained on the IM2GPS3k dataset)
        to produce a latitude / longitude heatmap from which the most likely
        named region is derived.

        Args:
            image_paths: List of relative paths to analyse.

        Returns:
            Human-readable region name (e.g. ``"Central Europe, Poland"``),
            or ``None`` if the model cannot determine a region with sufficient
            confidence.

        Example return value::

            "Central Europe, Poland"
        """
        logger.debug(
            "GeoVerificationAnalyzer.predict_region (MOCK): paths=%s", image_paths
        )
        # TODO: Replace with real geolocation CNN inference.
        return "Central Europe, Poland"  # Mock prediction
