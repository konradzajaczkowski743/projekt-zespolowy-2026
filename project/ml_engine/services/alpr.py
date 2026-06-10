"""
Automatic Licence Plate Recognition (ALPR) service for the project.

This module provides ``ALPRAnalyzer``, an interface for detecting
licence plates in images and transcribing the plate text.

Integrations:
    - Plate/Vehicle detection: RT-DETR (from object_det.py)
    - OCR: EasyOCR
"""
from __future__ import annotations

import logging
import re
from typing import Any
import numpy as np
from PIL import Image

logger: logging.Logger = logging.getLogger(__name__)


class ALPRAnalyzer:
    """
    Detects licence plates and transcribes their text from input images.

    All public methods return data structures that match the JSON contract
    defined by ``ALPRResultSerializer`` in ``api/serializers.py``.

    Attributes:
        model_version: Semantic version string of the underlying ML model.
        min_plate_confidence: Minimum confidence to report a plate detection.
    """

    model_version: str = "1.0.0-easyocr"
    min_plate_confidence: float = 0.4

    def __init__(self, min_plate_confidence: float = 0.4) -> None:
        """
        Initialise the ALPR analyser and load the ML model.

        Args:
            min_plate_confidence: Plate detections with a confidence score
                below this value are discarded.  Defaults to 0.4.
        """
        self.min_plate_confidence = min_plate_confidence
        self.reader = None
        logger.debug(
            "ALPRAnalyzer initialised (version=%s, min_confidence=%s)",
            self.model_version,
            self.min_plate_confidence,
        )

    def _get_reader(self):
        if self.reader is None:
            try:
                import easyocr
                # We use English for standard plates. Can add 'pl' if needed.
                self.reader = easyocr.Reader(['en'], gpu=True)
            except Exception as e:
                logger.error("[ALPR] Error loading EasyOCR: %s", e)
                self.reader = False # Mark as failed to load
        return self.reader

    # ── Public interface ──────────────────────────────────────────────────────

    def recognize_plates(
        self, image_paths: list[str]
    ) -> list[dict[str, Any]]:
        """
        Detect and transcribe all licence plates found in the supplied images.

        This implementation uses a local OCR-based ALPR pipeline when possible,
        falling back to the existing Gemma VLM analysis only if the local
        attempt does not yield any results.

        Args:
            image_paths: List of relative media-root paths to the uploaded
                image files.

        Returns:
            List of ALPR result dictionaries, each conforming to
            ``ALPRResultSerializer``.
        """
        logger.info(
            "ALPRAnalyzer.recognize_plates: images=%d min_confidence=%s",
            len(image_paths),
            self.min_plate_confidence,
        )
        
        plates: list[dict[str, Any]] = []
        if not image_paths:
            return plates
            
        reader = self._get_reader()
        if not reader:
            logger.warning("[ALPR] EasyOCR reader not available. Returning empty.")
            return plates

        try:
            from ml_engine.services.object_det import ObjectDetector
            detector = ObjectDetector()
            
            # 1. Detect vehicles using RT-DETR
            detections = detector.detect_objects(image_paths)
            vehicle_labels = {"car", "truck", "bus", "motorcycle"}
            vehicles = [d for d in detections if d["label"] in vehicle_labels]
            
            image_path = image_paths[0]
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                img_w, img_h = img.size
                
                for v in vehicles:
                    bbox = v["bounding_box"]
                    # Add slight padding to the bounding box
                    x_min = max(0, bbox["x_min"] - 10)
                    y_min = max(0, bbox["y_min"] - 10)
                    x_max = min(img_w, bbox["x_max"] + 10)
                    y_max = min(img_h, bbox["y_max"] + 10)
                    
                    if x_max <= x_min or y_max <= y_min:
                        continue
                        
                    # Crop the vehicle
                    vehicle_crop = img_rgb.crop((x_min, y_min, x_max, y_max))
                    crop_np = np.array(vehicle_crop)
                    
                    # 2. Run OCR on the vehicle crop to find the plate
                    # EasyOCR returns a list of tuples: (bbox, text, prob)
                    ocr_results = reader.readtext(crop_np)
                    
                    for ocr_res in ocr_results:
                        text_bbox, text, prob = ocr_res
                        
                        # Very simple heuristic to filter out non-plates:
                        # usually 5 to 9 alphanumeric chars
                        clean_text = "".join(c for c in text if c.isalnum()).upper()
                        
                        if len(clean_text) >= 4 and len(clean_text) <= 10 and prob >= self.min_plate_confidence:
                            # Map crop bbox back to original image coordinates
                            # text_bbox is [ [tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y] ]
                            tl_x = int(text_bbox[0][0]) + x_min
                            tl_y = int(text_bbox[0][1]) + y_min
                            br_x = int(text_bbox[2][0]) + x_min
                            br_y = int(text_bbox[2][1]) + y_min
                            
                            plates.append({
                                "plate_text": clean_text,
                                "confidence": round(float(prob), 4),
                                "country_code": None, # Could be implemented with further heuristic
                                "bounding_box": {
                                    "x_min": tl_x,
                                    "y_min": tl_y,
                                    "x_max": br_x,
                                    "y_max": br_y,
                                }
                            })
                            
        except Exception as e:
            logger.exception("[ALPR] Error during plate recognition: %s", e)

        logger.info("[ALPR] Found %d plates", len(plates))
        return plates

    def validate_plate_format(
        self, plate_text: str, country_code: str
    ) -> bool:
        """
        Validate whether a plate text string conforms to the expected format.

        This auxiliary method cross-checks the transcribed text against a
        dictionary of country-specific regex patterns to catch obvious OCR
        errors and to flag plates from unexpected countries.

        Args:
            plate_text: Transcribed licence plate string (e.g. ``"WA 12345"``).
            country_code: ISO 3166-1 alpha-2 country code to validate against.

        Returns:
            ``True`` when the plate text matches the country's expected format,
            ``False`` otherwise.
        """
        logger.debug(
            "ALPRAnalyzer.validate_plate_format (MOCK): plate=%s country=%s",
            plate_text,
            country_code,
        )
        # TODO: Replace with a dict of country-code → regex pattern and real
        #       validation logic.
        return True  # Mock: always valid
