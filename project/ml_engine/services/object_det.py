"""
Object detection service using Grounding DINO for the project.

This module provides ``ObjectDetector``, which uses the Grounding DINO
model for open-vocabulary object detection.

Model: IDEA-Research/grounding-dino-tiny (Apache 2.0 license)
    - Open-vocabulary detector
    - Can detect arbitrary objects based on text prompt
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


def _load_grounding_dino() -> tuple[Any, Any] | None:
    """Load Grounding DINO model with singleton cache."""
    cache_key = "grounding_dino"
    if cache_key not in _detector_cache:
        if not HAS_TORCH:
            return None

        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            repo_id = "IDEA-Research/grounding-dino-tiny"
            logger.info("[object_det] Loading Grounding DINO (%s)...", repo_id)

            processor = AutoProcessor.from_pretrained(
                repo_id, cache_dir=CACHE_DIR
            )
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                repo_id, cache_dir=CACHE_DIR
            )
            device = _get_device()
            if device is not None:
                model = model.to(device)
            model.eval()

            _detector_cache[f"{cache_key}_processor"] = processor
            _detector_cache[cache_key] = model
            logger.info("[object_det] Grounding DINO loaded successfully.")
        except Exception as exc:
            logger.exception("[object_det] Error loading Grounding DINO: %s", exc)
            _detector_cache[cache_key] = None
            _detector_cache[f"{cache_key}_processor"] = None

    proc = _detector_cache.get(f"{cache_key}_processor")
    mdl = _detector_cache.get(cache_key)
    if proc is None or mdl is None:
        return None
    return proc, mdl


class ObjectDetector:
    """
    Detects and classifies objects present in one or more images using Grounding DINO.

    All public methods return data structures that match the JSON contract
    defined by ``DetectedObjectSerializer`` in ``api/serializers.py``.

    Attributes:
        model_version: Semantic version string of the underlying ML model.
        confidence_threshold: Minimum confidence required to report a detection.
    """

    model_version: str = "2.0.0-grounding-dino"
    confidence_threshold: float = 0.35
    SCENERY_PROMPT: str = "castle . tower . church . building . monument . house . tree . forest . mountain . lake . river . cloud . sky . car . person . bicycle . dog . cat . horse . bridge . road . sign . window . roof ."

    def __init__(self, confidence_threshold: float = 0.35) -> None:
        """
        Initialise the detector.

        Args:
            confidence_threshold: Detections with a confidence below this
                value are discarded.  Defaults to 0.35.
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

        Uses Grounding DINO for open-vocabulary detection based on SCENERY_PROMPT.

        Args:
            image_paths: List of paths to the uploaded image files.

        Returns:
            List of detection dictionaries.
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

        bundle = _load_grounding_dino()
        if bundle is None:
            logger.warning("[object_det] Grounding DINO model unavailable — returning empty")
            return []

        processor, model = bundle
        device = _get_device()

        try:
            inputs = processor(images=image, text=self.SCENERY_PROMPT, return_tensors="pt")
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.confidence_threshold,
                text_threshold=0.3,
                target_sizes=target_sizes,
            )[0]

            detections: list[dict[str, Any]] = []

            for score, label_name, box in zip(
                results["scores"].tolist(),
                results["labels"],
                results["boxes"].tolist(),
            ):
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
            logger.exception("[object_det] Grounding DINO inference error: %s", e)
            return []

    def classify_scene(self, image_paths: list[str]) -> dict[str, float]:
        """
        Classify the overall scene depicted in the images using Zero-Shot StreetCLIP.
        
        Args:
            image_paths: List of relative paths to analyse.

        Returns:
            Dictionary mapping scene-category labels to confidence scores.
        """
        logger.debug(
            "ObjectDetector.classify_scene: paths=%s", image_paths
        )
        
        if not image_paths or not HAS_TORCH:
            return {"outdoor": 0.5, "urban": 0.5, "daylight": 0.5}
            
        try:
            image = Image.open(image_paths[0]).convert("RGB")
            from ml_engine.services.geolocator import _classify_with_streetclip
            
            labels = ["indoor", "outdoor", "urban", "nature", "night", "daylight", "beach", "forest"]
            results = _classify_with_streetclip(image, candidate_labels=labels, top_k=len(labels))
            
            scene_scores = {}
            for res in results:
                scene_scores[res["label"]] = round(float(res["score"]), 4)
                
            return scene_scores
            
        except Exception as e:
            logger.exception("[object_det] Scene classification error: %s", e)
            return {"outdoor": 0.5, "urban": 0.5, "daylight": 0.5}
