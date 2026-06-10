"""
Object detection service using RT-DETR for the project.

This module provides ``ObjectDetector``, which uses the RT-DETR
(Real-Time DEtection TRansformer) model for detecting and classifying
objects in images.

Model: PekingU/rtdetr_r50vd (Apache 2.0 license)
    - Real-time transformer-based detector
    - COCO-pretrained (80 object categories)
    - Runs locally without external APIs
"""
from __future__ import annotations

import logging
import os
from typing import Any

from PIL import Image

logger: logging.Logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("[object_det] PyTorch not installed — object detection disabled")

CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", "/tmp/models")
_detector_cache: dict[str, Any] = {}


def _get_device() -> Any:
    """Return the best available torch device."""
    if not HAS_TORCH:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_rtdetr() -> tuple[Any, Any] | None:
    """Load RT-DETR model with singleton cache."""
    cache_key = "rtdetr"
    if cache_key not in _detector_cache:
        if not HAS_TORCH:
            return None

        try:
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

            repo_id = "PekingU/rtdetr_r50vd"
            logger.info("[object_det] Loading RT-DETR (%s)...", repo_id)

            processor = RTDetrImageProcessor.from_pretrained(
                repo_id, cache_dir=CACHE_DIR
            )
            model = RTDetrForObjectDetection.from_pretrained(
                repo_id, cache_dir=CACHE_DIR
            )
            device = _get_device()
            if device is not None:
                model = model.to(device)
            model.eval()

            _detector_cache[f"{cache_key}_processor"] = processor
            _detector_cache[cache_key] = model
            logger.info("[object_det] RT-DETR loaded successfully.")
        except Exception as exc:
            logger.exception("[object_det] Error loading RT-DETR: %s", exc)
            _detector_cache[cache_key] = None
            _detector_cache[f"{cache_key}_processor"] = None

    proc = _detector_cache.get(f"{cache_key}_processor")
    mdl = _detector_cache.get(cache_key)
    if proc is None or mdl is None:
        return None
    return proc, mdl


class ObjectDetector:
    """
    Detects and classifies objects present in one or more images using RT-DETR.

    All public methods return data structures that match the JSON contract
    defined by ``DetectedObjectSerializer`` in ``api/serializers.py``.

    Attributes:
        model_version: Semantic version string of the underlying ML model.
        confidence_threshold: Minimum confidence required to report a detection.
    """

    model_version: str = "1.0.0-rtdetr"
    confidence_threshold: float = 0.5

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        """
        Initialise the detector.

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

        Uses RT-DETR (PekingU/rtdetr_r50vd) for real-time object detection.
        Falls back to an empty list if the model is unavailable.

        Args:
            image_paths: List of paths to the uploaded image files.

        Returns:
            List of detection dictionaries::

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

        if not image_paths or not HAS_TORCH:
            logger.warning("[object_det] No images or PyTorch unavailable")
            return []

        image_path = image_paths[0]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error("[object_det] Error reading image: %s", e)
            return []

        bundle = _load_rtdetr()
        if bundle is None:
            logger.warning("[object_det] RT-DETR model unavailable — returning empty")
            return []

        processor, model = bundle
        device = _get_device()

        try:
            inputs = processor(images=image, return_tensors="pt")
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process detections
            target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
            results = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.confidence_threshold,
            )[0]

            detections: list[dict[str, Any]] = []

            for score, label_id, box in zip(
                results["scores"].tolist(),
                results["labels"].tolist(),
                results["boxes"].tolist(),
            ):
                label_name = model.config.id2label.get(label_id, f"class_{label_id}")
                x_min, y_min, x_max, y_max = [int(coord) for coord in box]

                detections.append({
                    "label": label_name,
                    "confidence": round(score, 4),
                    "bounding_box": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                    },
                })

            logger.info(
                "ObjectDetector.detect_objects: found %d objects above threshold",
                len(detections),
            )
            return detections

        except Exception as e:
            logger.exception("[object_det] RT-DETR inference error: %s", e)
            return []

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
