"""
Automatic Licence Plate Recognition (ALPR) service stub for the project.

This module provides ``ALPRAnalyzer``, a clean interface (mock) for detecting
licence plates in images and transcribing the plate text.

Expected real-model integrations (to be added later):
    - Permissive two-stage pipeline: plate detection (e.g., RT-DETR, Faster R-CNN)
      followed by OCR (EasyOCR, TrOCR, or PaddleOCR under Apache 2.0).
    - Fully local country-of-origin classification based on plate font and layout
      using a custom ResNet or MobileNet classifier.
    - No external APIs or restrictive/AGPL models (like Ultralytics YOLO) are allowed.
"""
from __future__ import annotations

import logging
import re
from typing import Any

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

    model_version: str = "0.1.0-mock"
    min_plate_confidence: float = 0.6

    def __init__(self, min_plate_confidence: float = 0.6) -> None:
        """
        Initialise the ALPR analyser and (eventually) load the ML model.

        Args:
            min_plate_confidence: Plate detections with a confidence score
                below this value are discarded.  Defaults to 0.6.
        """
        self.min_plate_confidence = min_plate_confidence
        logger.debug(
            "ALPRAnalyzer initialised (version=%s, min_confidence=%s)",
            self.model_version,
            self.min_plate_confidence,
        )

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

        from os import getenv

        if getenv("ALPR_MOCK", "false").lower() in ("1", "true", "yes"):
            mock_plates: list[dict[str, Any]] = [
                {
                    "plate_text": "WA 12345",
                    "confidence": 0.94,
                    "country_code": "PL",
                    "bounding_box": {
                        "x_min": 430,
                        "y_min": 390,
                        "x_max": 620,
                        "y_max": 440,
                    },
                }
            ]
            logger.debug(
                "ALPRAnalyzer.recognize_plates (MOCK): returning %d plate(s)",
                len(mock_plates),
            )
            return mock_plates

        results = []
        try:
            results = self._recognize_with_easyocr(image_paths)
            if results:
                return results
        except Exception as exc:
            logger.exception("ALPRAnalyzer.recognize_plates easyocr failed: %s", exc)

        # Fallback: try using the existing Gemma VLM if easyocr produced no plates.
        try:
            from ml_engine.services.image_description import ImageDescriptionAnalyzer
            import json
            import re

            analyzer = ImageDescriptionAnalyzer()
            results = []

            for path in image_paths:
                prompt = (
                    "Proszę zlokalizuj wszystkie tablice rejestracyjne na tym obrazie i zwróć WYŁĄCZNIE JSON w formacie listy obiektów, np.:\n\n"
                    "[\n"
                    "  {\n"
                    "    \"plate_text\": \"ABC 1234\",\n"
                    "    \"confidence\": 0.95,\n"
                    "    \"country_code\": \"PL\",\n"
                    "    \"bounding_box\": {\"x_min\": 100, \"y_min\": 200, \"x_max\": 300, \"y_max\": 240}\n"
                    "  }\n"
                    "]\n\n"
                    "Jeżeli nie ma tablic, zwróć pustą listę: []\n"
                )

                raw = analyzer.query_image_with_context(path, prompt)
                m = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
                if not m:
                    m2 = re.search(r"\{.*\}", raw, re.DOTALL)
                    if m2:
                        try:
                            obj = json.loads(m2.group(0))
                            if isinstance(obj, dict):
                                results.append(obj)
                        except json.JSONDecodeError:
                            logger.debug("ALPR fallback: could not parse single-object JSON from Gemma")
                    else:
                        logger.debug("ALPR fallback: no JSON array found in Gemma response for %s", path)
                else:
                    try:
                        arr = json.loads(m.group(0))
                        if isinstance(arr, list):
                            results.extend(arr)
                    except json.JSONDecodeError:
                        logger.debug("ALPR fallback: JSON array parse failed for Gemma response")

            clean: list[dict[str, Any]] = []
            for r in results:
                if not isinstance(r, dict):
                    continue
                plate_text = r.get("plate_text")
                confidence = float(r.get("confidence", 0.0)) if r.get("confidence") is not None else 0.0
                bbox = r.get("bounding_box") or {}
                if plate_text and isinstance(bbox, dict):
                    plate_text_clean = str(plate_text).strip()
                    if not self.validate_plate_format(plate_text_clean, r.get("country_code") or "PL"):
                        continue
                    clean.append({
                        "plate_text": plate_text_clean,
                        "confidence": round(confidence, 4),
                        "country_code": r.get("country_code"),
                        "bounding_box": {
                            "x_min": int(bbox.get("x_min", 0)),
                            "y_min": int(bbox.get("y_min", 0)),
                            "x_max": int(bbox.get("x_max", 0)),
                            "y_max": int(bbox.get("y_max", 0)),
                        },
                    })

            if clean:
                logger.debug("ALPRAnalyzer.recognize_plates: Gemma fallback returned %d plate(s)", len(clean))
                return clean
            logger.debug("ALPRAnalyzer.recognize_plates: Gemma fallback found no plates")
            return []
        except Exception as e:
            logger.exception("ALPRAnalyzer.recognize_plates: fallback via Gemma failed: %s", e)
            return []

    def _recognize_with_easyocr(self, image_paths: list[str]) -> list[dict[str, Any]]:
        try:
            import easyocr  # type: ignore[import]
        except ImportError:
            logger.info("EasyOCR is not installed; skipping local ALPR.")
            return []

        logger.debug("ALPRAnalyzer._recognize_with_easyocr: using EasyOCR for local plate recognition")
        reader = easyocr.Reader(["pl", "en"], gpu=False)
        seen: dict[str, dict[str, Any]] = {}

        for path in image_paths:
            try:
                raw_results = reader.readtext(path, detail=1, paragraph=False)
            except Exception as exc:
                logger.exception("ALPRAnalyzer._recognize_with_easyocr: OCR failed for %s: %s", path, exc)
                continue

            for bbox, text, confidence in raw_results:
                if confidence < self.min_plate_confidence:
                    continue

                cleaned_text = re.sub(r"[^A-Z0-9 ]", "", text.upper()).strip()
                cleaned_text = cleaned_text.replace(" ", "")
                if not cleaned_text:
                    continue

                country_code = self._infer_country_code(cleaned_text)
                if not self.validate_plate_format(cleaned_text, country_code or "PL"):
                    continue

                x_coords = [int(point[0]) for point in bbox]
                y_coords = [int(point[1]) for point in bbox]
                plate_entry = {
                    "plate_text": cleaned_text,
                    "confidence": round(float(confidence), 4),
                    "country_code": country_code,
                    "bounding_box": {
                        "x_min": min(x_coords),
                        "y_min": min(y_coords),
                        "x_max": max(x_coords),
                        "y_max": max(y_coords),
                    },
                }

                existing = seen.get(cleaned_text)
                if existing is None or existing["confidence"] < plate_entry["confidence"]:
                    seen[cleaned_text] = plate_entry

        return list(seen.values())

    def _infer_country_code(self, plate_text: str) -> str | None:
        if re.fullmatch(r"[A-Z]{1,3}[0-9]{1,5}", plate_text):
            return "PL"
        if re.fullmatch(r"[A-Z0-9]{5,10}", plate_text):
            return "PL"
        return None

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
        if country_code == "PL":
            return bool(re.fullmatch(r"[A-Z]{1,3}[0-9]{1,5}", plate_text))
        return bool(re.fullmatch(r"[A-Z0-9]{5,10}", plate_text))

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
