"""
Object detection service stub for the project.

This module provides ``ObjectDetector``, a clean interface (mock) for detecting
and classifying objects in one or more images.

Expected real-model integrations (to be added later):
    - RT-DETR (Real-Time DEtection TRansformer) via Hugging Face ``transformers``
      (Apache 2.0 license).
    - Faster R-CNN or SSD via ``torchvision`` (BSD 3-Clause license).
    - Custom fine-tuned local model for forensic-specific object categories
      (weapons, documents, vehicles, faces, ...).
    - No external APIs or restrictive/AGPL models (like Ultralytics YOLO) are allowed.
"""
from __future__ import annotations

import logging
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Detects and classifies objects present in one or more images.

    All public methods return data structures that match the JSON contract
    defined by ``DetectedObjectSerializer`` in ``api/serializers.py``.

    Attributes:
        model_version: Semantic version string of the underlying ML model.
        confidence_threshold: Minimum confidence required to report a detection.
    """

    model_version: str = "0.1.0-mock"
    confidence_threshold: float = 0.5

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        """
        Initialise the detector and (eventually) load the ML model.

        Args:
            confidence_threshold: Detections with a confidence below this
                value are discarded.  Defaults to 0.5.
        """
        self.confidence_threshold = confidence_threshold
        logger.debug(
            "ObjectDetector initialised (version=%s, threshold=%s)",
            self.model_version,
            self.confidence_threshold,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def detect_objects(
        self, image_paths: list[str]
    ) -> list[dict[str, Any]]:
        """
        Detect all objects in the supplied images above the confidence threshold.

        In production this method will load each image, run it through the
        permissive local model (e.g., RT-DETR, Faster R-CNN), apply Non-Maximum
        results by ``confidence_threshold``.

        Args:
            image_paths: List of relative media-root paths to the uploaded
                image files.

        Returns:
            List of detection dictionaries, each conforming to
            ``DetectedObjectSerializer``::

                [
                    {
                        "label": str,
                        "confidence": float,     # [0.0, 1.0]
                        "bounding_box": {
                            "x_min": int,
                            "y_min": int,
                            "x_max": int,
                            "y_max": int
                        }
                    },
                    ...
                ]

            Returns an empty list when no objects meet the confidence threshold.
        """
        logger.info(
            "ObjectDetector.detect_objects: images=%d threshold=%s",
            len(image_paths),
            self.confidence_threshold,
        )
        # TODO: Replace with real permissive inference logic (e.g., RT-DETR).
        mock_detections: list[dict[str, Any]] = [
            {
                "label": "person",
                "confidence": 0.92,
                "bounding_box": {"x_min": 120, "y_min": 45, "x_max": 310, "y_max": 480},
            },
            {
                "label": "vehicle",
                "confidence": 0.87,
                "bounding_box": {"x_min": 400, "y_min": 200, "x_max": 750, "y_max": 520},
            },
        ]
        logger.debug(
            "ObjectDetector.detect_objects (MOCK): returning %d detections",
            len(mock_detections),
        )
        return mock_detections

    def classify_scene(self, image_paths: list[str]) -> dict[str, float]:
        """
        Classify the overall scene depicted in the images.

        This auxiliary method complements ``detect_objects`` by providing a
        high-level scene label (e.g. ``"outdoor"`` / ``"indoor"`` /
        ``"urban"``), useful as a feature for the geo-verification pipeline.

        Args:
            image_paths: List of relative paths to analyse.

        Returns:
            Dictionary mapping scene-category labels to confidence scores.

        Example return value::

            {"outdoor": 0.78, "urban": 0.61, "daylight": 0.93}
        """
        logger.debug(
            "ObjectDetector.classify_scene (MOCK): paths=%s", image_paths
        )
        # TODO: Replace with real scene-classification model inference.
        return {"outdoor": 0.78, "urban": 0.61, "daylight": 0.93}
