"""
Distance estimation service using Depth Anything V2 for the project.

This module provides ``DistanceEstimator`` which combines monocular depth
estimation (Depth Anything V2 Base) with object detection results to estimate
the distance from the camera to the main object in an image.

Key features:
    - Determines whether a main object exists (vs. landscape/nature scenes)
    - Uses relative depth map + object-size heuristics for distance approximation
    - Returns null distance when no distinct main object is present (e.g. beach, sky)

Model: depth-anything/Depth-Anything-V2-Base-hf (Apache 2.0 license)
    - State-of-the-art monocular depth estimation
    - ~400MB, runs locally on CPU
    - Outputs relative depth map (not metric) — heuristics convert to metres
"""
from __future__ import annotations

import logging
import json
import difflib
import math
import os
import re
from typing import Any

import numpy as np
import requests
from PIL import Image

logger: logging.Logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("[distance] PyTorch not installed — depth estimation disabled")

CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", "/tmp/models")
_depth_cache: dict[str, Any] = {}

# Load local landmarks DB
_local_landmarks_db = []
try:
    _db_path = os.path.join(os.path.dirname(__file__), "..", "data", "polish_landmarks_db.json")
    with open(_db_path, "r", encoding="utf-8") as f:
        _local_landmarks_db = json.load(f)
except Exception as e:
    logger.warning("[distance] Failed to load local DB: %s", e)

# Typical real-world sizes (height in metres) for common COCO categories.
# Used to calibrate relative depth → absolute distance.
OBJECT_SIZES_M: dict[str, float] = {
    "person": 1.70,
    "bicycle": 1.00,
    "car": 1.50,
    "motorcycle": 1.10,
    "bus": 3.00,
    "truck": 3.50,
    "train": 4.00,
    "boat": 2.50,
    "traffic light": 0.80,
    "fire hydrant": 0.50,
    "stop sign": 0.75,
    "bench": 0.80,
    "bird": 0.20,
    "cat": 0.30,
    "dog": 0.50,
    "horse": 1.60,
    "cow": 1.40,
    "elephant": 3.00,
    "bear": 1.50,
    "chair": 0.90,
    "couch": 0.85,
    "dining table": 0.75,
    "tv": 0.50,
    "laptop": 0.25,
    "cell phone": 0.15,
    "bottle": 0.25,
    "cup": 0.12,
    "umbrella": 1.00,
    "suitcase": 0.60,
    "sports ball": 0.22,
    # Landmark / building sizes (used when geo identifies a structure)
    "landmark": 40.0,
    "building": 25.0,
    "monument": 15.0,
    "church": 30.0,
    "tower": 50.0,
    "palace": 35.0,
    "castle": 30.0,
    "bridge": 20.0,
    "statue": 10.0,
}

# Keywords in landmark names that hint at the structure type for size estimation.
_LANDMARK_TYPE_HINTS: dict[str, str] = {
    "pałac": "palace", "palace": "palace",
    "wieża": "tower", "tower": "tower", "torre": "tower",
    "kościół": "church", "church": "church", "cathedral": "church",
    "katedra": "church", "basilica": "church", "bazylika": "church",
    "zamek": "castle", "castle": "castle",
    "most": "bridge", "bridge": "bridge", "pont": "bridge",
    "pomnik": "statue", "statue": "statue", "monument": "monument",
}

# Scene types from Places365 that indicate no main object
LANDSCAPE_SCENES: set[str] = {
    "beach", "coast", "ocean", "sea", "sky", "mountain", "valley",
    "field", "forest", "desert", "lake", "river", "waterfall",
    "horizon", "sunset", "sunrise", "landscape", "countryside",
    "tundra", "glacier", "prairie", "meadow", "canyon", "cliff",
    "archipelago", "islet", "marsh", "swamp", "badlands",
}


def _get_device() -> Any:
    """Return the best available torch device."""
    if not HAS_TORCH:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_depth_model() -> tuple[Any, Any] | None:
    """Load Depth Anything V2 Base with singleton cache."""
    cache_key = "depth_anything"
    if cache_key not in _depth_cache:
        if not HAS_TORCH:
            return None

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            repo_id = "depth-anything/Depth-Anything-V2-Base-hf"
            logger.info("[distance] Loading Depth Anything V2 Base (%s)...", repo_id)

            processor = AutoImageProcessor.from_pretrained(
                repo_id, cache_dir=CACHE_DIR
            )
            model = AutoModelForDepthEstimation.from_pretrained(
                repo_id, cache_dir=CACHE_DIR
            )
            device = _get_device()
            if device is not None:
                model = model.to(device)
            model.eval()

            _depth_cache[f"{cache_key}_processor"] = processor
            _depth_cache[cache_key] = model
            logger.info("[distance] Depth Anything V2 Base loaded.")
        except Exception as exc:
            logger.exception("[distance] Error loading Depth Anything V2: %s", exc)
            _depth_cache[cache_key] = None
            _depth_cache[f"{cache_key}_processor"] = None

    proc = _depth_cache.get(f"{cache_key}_processor")
    mdl = _depth_cache.get(cache_key)
    if proc is None or mdl is None:
        return None
    return proc, mdl


class DistanceEstimator:
    """
    Estimates the distance to the main object in an image.

    Pipeline:
        1. Generate depth map using Depth Anything V2 Base
        2. If detected objects exist → find the "main object" (largest + highest confidence)
        3. Extract median depth value in the main object's bounding box
        4. Convert relative depth to approximate metres using known object sizes
        5. If no objects detected (landscape/beach) → return has_main_object=False

    Attributes:
        model_version: Version of the depth estimation model.
    """

    model_version: str = "1.0.0-depth-anything-v2-base"

    def __init__(self) -> None:
        """Initialise the distance estimator."""
        logger.debug(
            "DistanceEstimator initialised (version=%s)", self.model_version
        )

    def estimate_distance(
        self,
        image_paths: list[str],
        detected_objects: list[dict[str, Any]],
        scene_type: str | None = None,
        geo_verification: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Estimate the distance to the main object in the image.

        When ``geo_verification`` identifies a landmark or building, the
        estimator treats the dominant central structure as the main object
        instead of relying solely on COCO detections (which lack building
        categories).

        Args:
            image_paths: Paths to image files.
            detected_objects: Output from ObjectDetector.detect_objects().
            scene_type: Optional scene classification (from places365 or geo).
            geo_verification: Optional geo-verification result dict.  When
                present and a landmark is identified, the estimator builds
                a virtual "landmark" object covering the central region of
                the image.

        Returns:
            Dictionary with distance estimation results.
        """
        logger.info(
            "DistanceEstimator.estimate_distance: images=%d objects=%d scene=%s geo=%s",
            len(image_paths),
            len(detected_objects),
            scene_type,
            bool(geo_verification),
        )

        # Default result for when estimation is not possible
        no_result: dict[str, Any] = {
            "has_main_object": False,
            "main_object": None,
            "reason": None,
            "depth_map_stats": None,
            "model_used": self.model_version,
        }

        if not image_paths or not HAS_TORCH:
            no_result["reason"] = "no_image_or_torch"
            return no_result

        # Check if this is a landscape scene (no distinct main object)
        # BUT: if geo identified a landmark, it's NOT a landscape
        has_landmark = self._has_geo_landmark(geo_verification)
        if not has_landmark and self._is_landscape_scene(detected_objects, scene_type):
            no_result["reason"] = "landscape_scene_no_main_object"
            logger.info("[distance] Landscape scene detected — no main object")
            return no_result

        # Open image for depth map
        image_path = image_paths[0]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error("[distance] Error reading image: %s", e)
            no_result["reason"] = "image_read_error"
            return no_result

        depth_map = self._generate_depth_map(image)
        if depth_map is None:
            no_result["reason"] = "depth_model_unavailable"
            return no_result

        # Determine main object: prefer landmark from geo, fallback to COCO
        if has_landmark:
            main_object = self._build_landmark_object(geo_verification, image.size, depth_map, image_path)
            logger.info(
                "[distance] Using landmark from geo as main object: %s",
                main_object["label"],
            )
        else:
            main_object = self._find_main_object(detected_objects)

        if main_object is None:
            no_result["reason"] = "no_objects_detected"
            logger.info("[distance] No objects detected — no main object")
            return no_result

        # Extract depth at the main object's bounding box
        bbox = main_object["bounding_box"]
        obj_depth = self._get_object_depth(depth_map, bbox, image.size)

        # Compute depth map statistics
        depth_stats = {
            "min_depth": round(float(np.min(depth_map)), 4),
            "max_depth": round(float(np.max(depth_map)), 4),
            "mean_depth": round(float(np.mean(depth_map)), 4),
        }

        # Estimate absolute distance using object-size heuristic
        estimated_distance_m = self._relative_to_metres(
            obj_depth,
            main_object["label"],
            bbox,
            image.size,
            depth_map,
            geo_verification,
        )

        result: dict[str, Any] = {
            "has_main_object": True,
            "main_object": {
                "label": main_object["label"],
                "confidence": main_object["confidence"],
                "bounding_box": bbox,
                "estimated_distance_m": (
                    round(estimated_distance_m, 2)
                    if estimated_distance_m is not None
                    else None
                ),
                "depth_value": round(float(obj_depth), 4),
            },
            "reason": None,
            "depth_map_stats": depth_stats,
            "model_used": self.model_version,
        }

        logger.info(
            "[distance] Main object: %s, depth=%.4f, distance=%.2fm",
            main_object["label"],
            obj_depth,
            estimated_distance_m if estimated_distance_m else 0,
        )
        return result

    def _is_landscape_scene(
        self,
        detected_objects: list[dict[str, Any]],
        scene_type: str | None,
    ) -> bool:
        """
        Determine whether the image depicts a landscape with no main object.

        Returns True for scenes like beaches, mountains, open fields, etc.
        """
        # If scene classification says landscape
        if scene_type:
            scene_lower = scene_type.lower()
            for keyword in LANDSCAPE_SCENES:
                if keyword in scene_lower:
                    return True

        # If no objects detected with decent confidence
        high_conf_objects = [
            o for o in detected_objects if o.get("confidence", 0) > 0.5
        ]
        if len(high_conf_objects) == 0:
            return True

        return False

    @staticmethod
    def _has_geo_landmark(geo: dict[str, Any] | None) -> bool:
        """Return True if geo-verification identified a landmark or building."""
        if not geo:
            return False
        # Vector DB match is our most reliable source
        if geo.get("prediction_source") == "vector_db":
            return True
        # If gemma arbitrated a valid location, assume it's a structural landmark/location
        if geo.get("prediction_source") == "gemma_arbitration" and geo.get("predicted_region") != "UNKNOWN":
            return True
        # StreetCLIP top prediction with decent confidence
        top5 = geo.get("streetclip_top5") or []
        if top5 and top5[0].get("score", 0) > 0.15:
            return True
        # ViT landmark with decent confidence
        region = geo.get("predicted_region", "") or ""
        confidence = geo.get("confidence_score", 0) or 0
        if "landmark" in region.lower() and confidence > 0.4:
            return True
        return False

    @staticmethod
    def _build_landmark_object(
        geo: dict[str, Any],
        image_size: tuple[int, int],
        depth_map: np.ndarray,
        image_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Build a virtual "landmark" detection from geo-verification data.

        Uses the depth map to estimate the bounding box of the building.
        Looks for a connected component in the center of the image.
        Falls back to central 60% if heuristic fails.
        """
        img_w, img_h = image_size

        # Determine landmark name: prefer Vector DB, then Gemma arbitration, then ViT, then StreetCLIP
        prediction_source = geo.get("prediction_source", "")
        vit_landmark = geo.get("vit_predicted_region", "")
        
        if prediction_source == "vector_db" and geo.get("predicted_region"):
            landmark_name = geo.get("predicted_region")
            confidence = geo.get("confidence_score", 0.8)
        elif prediction_source == "gemma_arbitration" and geo.get("predicted_region") and geo.get("predicted_region") != "UNKNOWN":
            landmark_name = geo.get("predicted_region")
            confidence = geo.get("confidence_score", 0.9)
        elif prediction_source == "vit_landmarks" and vit_landmark and "landmark:" in vit_landmark.lower():
            landmark_name = vit_landmark.split(":", 1)[1].strip()
            confidence = geo.get("confidence_score", 0.5)
        else:
            top5 = geo.get("streetclip_top5") or []
            if top5:
                landmark_name = top5[0].get("label", "landmark")
                confidence = top5[0].get("score", 0.5)
            else:
                landmark_name = vit_landmark or "landmark"
                confidence = geo.get("confidence_score", 0.5)

        # If the landmark name looks like a city (e.g. "Paris, France") and we have the image,
        # ask Gemma if there is a specific landmark visible.
        if "," in landmark_name and image_path:
            specific_landmark = DistanceEstimator._ask_gemma_for_landmark_in_city(image_path, landmark_name)
            if specific_landmark:
                landmark_name = specific_landmark
                confidence = 0.5  # Modest confidence for Gemma's fallback guess

        # Infer structure type from name for size heuristic
        label = "landmark"  # default
        name_lower = landmark_name.lower()
        for keyword, struct_type in _LANDMARK_TYPE_HINTS.items():
            if keyword in name_lower:
                label = struct_type
                break

        bbox = None
        if depth_map is not None:
            h, w = depth_map.shape
            
            # Heuristic: Find prominent tall structure (landmark) using depth profile
            # 1. Identify sky/background depth (usually at the top of the image)
            top_region = depth_map[0:int(h * 0.05), :]
            sky_depth = float(np.median(top_region))
            
            # 2. Foreground mask: pixels significantly closer than sky
            # Depth Anything V2: larger values = closer (0.0 is far, 1.0 is close)
            if sky_depth < 0.5:
                foreground_mask = depth_map > (sky_depth + 0.15)
            else:
                foreground_mask = depth_map < (sky_depth - 0.15)
            
            # Ensure we have enough foreground to analyze
            if np.sum(foreground_mask) > (w * h * 0.02):
                # 3. Project foreground mass onto X-axis
                x_profile = np.sum(foreground_mask, axis=0)
                
                # The landmark is likely the most massive/tallest structure -> peak in X profile
                peak_x = int(np.argmax(x_profile))
                
                # Expand left and right to find structure width
                threshold = x_profile[peak_x] * 0.25
                x_min = peak_x
                while x_min > 0 and x_profile[x_min] > threshold:
                    x_min -= 1
                x_max = peak_x
                while x_max < w - 1 and x_profile[x_max] > threshold:
                    x_max += 1
                
                # 4. Find Y bounds (height of the structure) within this column range
                tower_mask = foreground_mask[:, x_min:x_max]
                y_indices = np.where(np.any(tower_mask, axis=1))[0]
                
                if len(y_indices) > 0:
                    y_min = int(np.min(y_indices))
                    y_max = h - 1  # Assume building goes to the ground/bottom
                    
                    # Sanity check: ensure bounding box is somewhat substantial
                    if (x_max - x_min) > w * 0.03 and (y_max - y_min) > h * 0.1:
                        scale_x = img_w / w
                        scale_y = img_h / h
                        bbox = {
                            "x_min": max(0, int(x_min * scale_x)),
                            "y_min": max(0, int(y_min * scale_y)),
                            "x_max": min(img_w, int(x_max * scale_x)),
                            "y_max": min(img_h, int(y_max * scale_y)),
                        }

        if bbox is None:
            # Fallback: Central 60% bounding box
            margin_x = int(img_w * 0.20)
            margin_y = int(img_h * 0.05)
            bbox = {
                "x_min": margin_x,
                "y_min": margin_y,
                "x_max": img_w - margin_x,
                "y_max": int(img_h * 0.85),
            }

        return {
            "label": f"{label} ({landmark_name})",
            "confidence": round(float(confidence), 4),
            "bounding_box": bbox,
        }

    @staticmethod
    def _find_main_object(
        detected_objects: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Select the "main object" from detected objects.

        Criteria: largest bounding box area × confidence score.
        """
        if not detected_objects:
            return None

        def score_object(obj: dict[str, Any]) -> float:
            bbox = obj.get("bounding_box", {})
            width = bbox.get("x_max", 0) - bbox.get("x_min", 0)
            height = bbox.get("y_max", 0) - bbox.get("y_min", 0)
            area = max(width * height, 1)
            confidence = obj.get("confidence", 0.0)
            return area * confidence

        return max(detected_objects, key=score_object)

    @staticmethod
    def _ask_gemma_for_landmark_in_city(image_path: str, city_name: str) -> str | None:
        """
        Dynamically ask Gemma LLM if there is a specific landmark in the image of a city.
        """
        import base64
        import os
        import requests

        gemma_api_url = os.getenv("GEMMA_API", "http://gemma:11434")
        model_name = os.getenv("MODEL_NAME", "gemma4:e2b")

        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            prompt = (
                f"This image was taken in {city_name}. Is there a specific, prominent landmark or "
                f"famous building visible in the image whose height is well-known? "
                f"If yes, respond ONLY with the name of that specific landmark. "
                f"If no, or if it is just a general city view without a main landmark, respond with 'NO'."
            )

            logger.info("[distance] Asking Gemma for specific landmark in: %s", city_name)
            response = requests.post(
                f"{gemma_api_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False,
                },
                timeout=120,
            )
            if response.status_code == 200:
                text = response.json().get("response", "").strip()
                if text and text.upper() != "NO" and len(text) < 50:
                    logger.info("[distance] Gemma identified specific landmark: %s", text)
                    return text
        except Exception as e:
            logger.warning("[distance] Gemma landmark query failed: %s", e)
            
        return None

    @staticmethod
    def _ask_gemma_for_height(landmark_name: str) -> float | None:
        """
        Dynamically ask Gemma LLM for the real-world height of a landmark.
        """
        import os
        gemma_api_url = os.getenv("GEMMA_API", "http://gemma:11434")
        model_name = os.getenv("MODEL_NAME", "gemma4:e2b")

        prompt = (
            f"What is the approximate height of {landmark_name} in meters? "
            f"Respond ONLY with a single integer representing the height in meters. "
            f"Do not add any text, units, or explanation."
        )

        try:
            logger.info("[distance] Asking Gemma for height of: %s", landmark_name)
            response = requests.post(
                f"{gemma_api_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )
            if response.status_code == 200:
                text = response.json().get("response", "").strip()
                # Extract first number found in the response
                match = re.search(r"(\d+)", text)
                if match:
                    height = float(match.group(1))
                    if 10.0 <= height <= 1000.0:  # Sanity check
                        logger.info("[distance] Gemma returned height: %sm", height)
                        return height
        except Exception as e:
            logger.warning("[distance] Gemma height query failed: %s", e)
            
        return None

    def _generate_depth_map(self, image: Image.Image) -> np.ndarray | None:
        """
        Generate a relative depth map using Depth Anything V2 Base.

        Returns a 2D numpy array where higher values = further from camera.
        """
        bundle = _load_depth_model()
        if bundle is None:
            return None

        processor, model = bundle
        device = _get_device()

        try:
            inputs = processor(images=image, return_tensors="pt")
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Interpolate to original image size
            depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],  # (height, width)
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_map = depth.cpu().numpy()

            # Normalize to [0, 1] range
            d_min = depth_map.min()
            d_max = depth_map.max()
            if d_max - d_min > 0:
                depth_map = (depth_map - d_min) / (d_max - d_min)

            return depth_map

        except Exception as e:
            logger.exception("[distance] Depth map generation failed: %s", e)
            return None

    @staticmethod
    def _get_object_depth(
        depth_map: np.ndarray,
        bbox: dict[str, int],
        image_size: tuple[int, int],
    ) -> float:
        """
        Extract the median depth value within an object's bounding box.

        Args:
            depth_map: Normalised depth map (H, W) in [0, 1].
            bbox: Bounding box with x_min, y_min, x_max, y_max.
            image_size: (width, height) of the original image.
        """
        h, w = depth_map.shape
        img_w, img_h = image_size

        # Scale bbox to depth map dimensions
        scale_x = w / img_w
        scale_y = h / img_h

        x_min = max(0, int(bbox["x_min"] * scale_x))
        y_min = max(0, int(bbox["y_min"] * scale_y))
        x_max = min(w, int(bbox["x_max"] * scale_x))
        y_max = min(h, int(bbox["y_max"] * scale_y))

        if x_max <= x_min or y_max <= y_min:
            return float(np.median(depth_map))

        region = depth_map[y_min:y_max, x_min:x_max]
        return float(np.median(region))

    def _relative_to_metres(
        self,
        relative_depth: float,
        label: str,
        bbox: dict[str, int],
        image_size: tuple[int, int],
        depth_map: np.ndarray,
        geo_verification: dict[str, Any] | None = None,
    ) -> float | None:
        """
        Convert relative depth to approximate distance in metres.

        Uses a heuristic based on:
        1. Known real-world size of the detected object type
        2. Apparent size (fraction of image height occupied by bbox)
        3. Typical camera field of view (~60° vertical)

        This is an approximation — not a precise measurement.
        """
        # Remove any bracketed info from label e.g., "landmark (Barcelona, Spain)" -> "landmark"
        base_label = label.split(" (")[0].strip()
        known_size = OBJECT_SIZES_M.get(base_label)
        
        # Hardcoded heights to prevent Gemma LLM hallucinations
        KNOWN_LANDMARK_HEIGHTS_M = {
            "sagrada familia": 172.0,
            "eiffel tower": 330.0,
            "wieża eiffla": 330.0,
            "pałac kultury": 237.0,
            "palace of culture": 237.0,
            "zamek królewski w warszawie": 60.0,
            "royal castle": 60.0,
            "sukiennice": 18.0,
            "kościół mariacki": 82.0,
            "st. mary's basilica": 82.0,
            "wawel": 50.0,
            "zamek w malborku": 50.0,
            "malbork castle": 50.0,
            "hala stulecia": 42.0,
            "spodek": 35.0,
            "zamek książ": 60.0,
            "big ben": 96.0,
            "colosseum": 48.0,
            "koloseum": 48.0,
            "statue of liberty": 93.0,
            "statua wolności": 93.0,
            "burj khalifa": 828.0,
            "taj mahal": 73.0,
            "empire state building": 381.0,
        }
        
        # If it's a landmark (identified by structural keywords or brackets), try hardcoded list first, then ask Gemma
        is_landmark = any(k in label.lower() for k in ["landmark", "building", "palace", "tower", "castle", "church", "monument", "bridge", "statue"])
        if is_landmark or ("(" in label and ")" in label):
            if "(" in label and ")" in label:
                landmark_name = label.split("(")[1].split(")")[0].strip()
                name_lower = landmark_name.lower()
                
                # Check hardcoded list
                for key, h in KNOWN_LANDMARK_HEIGHTS_M.items():
                    if key in name_lower:
                        known_size = h
                        logger.info("[distance] Used hardcoded height for %s: %sm", key, h)
                        break
                else:
                    # Check local JSON DB with fuzzy matching
                    best_match = None
                    best_ratio = 0.0
                    for _lm in _local_landmarks_db:
                        if not _lm.get("name"):
                            continue
                        ratio = difflib.SequenceMatcher(None, name_lower, _lm["name"].lower()).ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_match = _lm
                    
                    if best_match and best_ratio > 0.6 and best_match.get("height"):
                        known_size = float(best_match["height"])
                        logger.info("[distance] Used local DB height for %s (matched %s, ratio %.2f): %sm", landmark_name, best_match["name"], best_ratio, known_size)
                    else:
                        # Primary attempt
                        dynamic_height = self._ask_gemma_for_height(landmark_name)
                        
                        # Fallback attempt using ViT base landmark name if primary fails
                        if dynamic_height is None and geo_verification:
                            vit_landmark = geo_verification.get("vit_predicted_region", "")
                            if vit_landmark and "landmark:" in vit_landmark.lower():
                                fallback_name = vit_landmark.split(":", 1)[1].strip()
                                if fallback_name.lower() not in landmark_name.lower():
                                    logger.info("[distance] Fallback height query using ViT name: %s", fallback_name)
                                    dynamic_height = self._ask_gemma_for_height(fallback_name)
                        
                        if dynamic_height:
                            known_size = dynamic_height
        
        if known_size is None:
            # For unknown objects, use a rough mapping:
            # relative_depth 0.0 (close) → ~1m, 1.0 (far) → ~100m
            # Exponential scaling gives more realistic distribution
            if relative_depth < 0.01:
                return 0.5
            return round(0.5 + 99.5 * (relative_depth ** 1.5), 2)

        # Calculate apparent height as fraction of image
        img_w, img_h = image_size
        bbox_height = bbox.get("y_max", 0) - bbox.get("y_min", 0)
        apparent_fraction = bbox_height / max(img_h, 1)

        if apparent_fraction < 0.01:
            # Object is tiny in frame → very far
            return round(known_size / 0.01 * 2.0, 2)

        # Pinhole camera approximation:
        # distance ≈ (real_height * focal_length) / apparent_pixel_height
        # Assuming ~60° vertical FOV:
        # focal_length_px ≈ img_h / (2 * tan(30°)) ≈ img_h * 0.866
        focal_length_px = img_h * 0.866
        distance = (known_size * focal_length_px) / max(bbox_height, 1)

        # Blend with depth-map signal for better accuracy
        # If the depth map says the object is much closer/further, adjust
        depth_factor = 1.0 + (relative_depth - 0.5) * 0.4
        distance *= depth_factor

        return max(0.3, round(distance, 2))
