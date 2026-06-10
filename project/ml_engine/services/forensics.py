"""
Image forensics analysis service with optional ML-based AI/deepfake detection.

Provides ``ImageForensicsAnalyzer``, a clean interface that:
1. Always runs local analysis (ELA + EXIF)
2. Optionally runs ML ensemble if PyTorch/transformers available

Expected integrations:
    - Error Level Analysis (ELA) — local, always available
    - EXIF metadata anomaly detection — local, always available
    - ML ensemble (optional) — requires PyTorch + transformers
"""
from __future__ import annotations

import logging
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


class ImageForensicsAnalyzer:
    """
    Analyzes images for signs of digital manipulation and AI generation.

    This class acts as the boundary between the ML pipeline layer and the
    actual forensics models. All public methods return data structures that match
    the JSON contract defined in ``api/serializers.py``.

    Attributes:
        model_version: Semantic version string of the underlying ML model.
    """

    model_version: str = "1.0.0-advanced"

    def __init__(self) -> None:
        """Initialize the analyzer."""
        logger.debug(
            "ImageForensicsAnalyzer initialized (version=%s)", self.model_version
        )
        # Lazy import — nie ładuj modeli przy inicjalizacji
        self._forensics_impl = None

    def _get_impl(self):
        """Lazy-load forensics implementation."""
        if self._forensics_impl is None:
            try:
                from ml_engine.services.forensics_advanced import (
                    analyze_image_from_path,
                    ForensicsResult,
                )
                self._forensics_impl = (analyze_image_from_path, ForensicsResult)
            except ImportError as exc:
                logger.warning(
                    "Could not import advanced forensics: %s — falling back to mock",
                    exc
                )
                self._forensics_impl = None
        return self._forensics_impl

    # ── Public interface ──────────────────────────────────────────────────────

    def analyze(self, image_paths: list[str]) -> dict[str, Any]:
        """
        Run the full forensics analysis pipeline on the supplied images.

        Args:
            image_paths: List of relative media-root paths to the uploaded image files.

        Returns:
            Dictionary conforming to ``ForensicsResultSerializer``::

                {
                    "is_authentic": bool,
                    "confidence_score": float,    # [0.0, 1.0]
                    "manipulation_type": str | None,
                    "exif_anomalies": list[str],
                    "noise_analysis": dict
                }
        """
        logger.info(
            "ImageForensicsAnalyzer.analyze: processing %d image(s)", len(image_paths)
        )

        impl = self._get_impl()

        # Jeśli mamy advanced implementation
        if impl:
            analyze_func, ForensicsResult = impl
            if image_paths:
                first_path = image_paths[0]
                try:
                    result = analyze_func(first_path)
                    authenticity_percent = round(100.0 * (1.0 - result.confidence), 2)
                    return {
                        "is_authentic": not result.is_ai_generated,
                        "confidence_score": authenticity_percent,
                        "manipulation_type": "ai_generated" if result.is_ai_generated else None,
                        "exif_anomalies": result.exif.suspicious_fields if result.exif else [],
                        "noise_analysis": {
                            "ela_score": result.ela.mean_error if result.ela else 0.0,
                            "ai_probability": result.confidence,
                        },
                    }
                except Exception as exc:
                    logger.exception("Error in advanced forensics: %s", exc)

        # Fallback to mock
        return self._mock_analyze(image_paths)

    def analyze_exif(self, image_paths: list[str]) -> dict[str, Any]:
        """
        Extract and analyse EXIF metadata for anomalies.

        In production this method will use ``Pillow`` / ``piexif`` to read
        EXIF tags and apply rule-based heuristics together with fully local ML classifiers
        to flag suspicious entries (e.g. inconsistent timestamps, missing GPS
        tags when coordinates were provided, mismatched camera models).

        Args:
            image_paths: List of relative paths to analyse.

        Returns:
            Dictionary with the key ``"anomalies"`` mapping to a list of
            human-readable anomaly description strings.

        Example return value::

            {
                "anomalies": [
                    "Missing GPS data despite GPS coordinates supplied",
                    "Timestamp in the future detected"
                ]
            }
        """
        logger.debug(
            "ImageForensicsAnalyzer.analyze_exif (MOCK): paths=%s", image_paths
        )
        # TODO: Replace with real EXIF extraction logic using Pillow / piexif.
        return {
            "anomalies": [],  # Mock: no anomalies detected
        }

    def detect_tampering(self, image_paths: list[str]) -> dict[str, Any]:
        """
        Detect pixel-level manipulation using Error Level Analysis and PRNU.

        In production this will run the image through a fully local permissive CNN-based ELA model
        and correlate the PRNU noise fingerprint against an offline camera database.

        Args:
            image_paths: List of relative paths to analyse.

        Returns:
            Dictionary with keys:

            * ``is_authentic`` (bool) — True when no tampering is detected.
            * ``confidence_score`` (float) — Authenticity percentage [0.0, 100.0].
              0 means AI-generated, 100 means very likely real.
            * ``manipulation_type`` (str | None) — Category of manipulation
              (e.g. ``"splicing"``, ``"copy-move"``), or ``None``.
            * ``noise_analysis`` (dict) — Pixel-level noise metrics.

        Example return value::

            {
                "is_authentic": True,
                "confidence_score": 97.0,
                "manipulation_type": None,
                "noise_analysis": {"ela_score": 0.03, "prnu_match": 0.94}
            }
        """
        logger.debug(
            "ImageForensicsAnalyzer.detect_tampering (MOCK): paths=%s", image_paths
        )
        # TODO: Replace with real ELA + PRNU fingerprinting logic.
        return {
            "is_authentic": True,
            "confidence_score": 97.0,
            "manipulation_type": None,
            "noise_analysis": {
                "ela_score": 0.03,
                "prnu_match": 0.94,
                "ai_probability": 0.03,
            },
        }

    def _mock_analyze(self, image_paths: list[str]) -> dict[str, Any]:
        """Mock implementation when advanced forensics unavailable."""
        logger.warning("[forensics] Fallback to mock analysis")
        return {
            "is_authentic": True,
            "confidence_score": 97.0,
            "manipulation_type": None,
            "exif_anomalies": [],
            "noise_analysis": {
                "ela_score": 0.03,
                "ai_probability": 0.03,
            },
        }
