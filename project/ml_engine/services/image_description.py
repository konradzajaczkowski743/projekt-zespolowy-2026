"""
Image description analysis service using Gemma LLM.

This module provides ``ImageDescriptionAnalyzer``, which uses the Ollama API
to send images to the Gemma4 model for generating detailed descriptions of
image content (objects, people, scenes, etc.).

When an image is confirmed as authentic, this service provides a human-readable
summary of what the image contains.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any

import requests

logger: logging.Logger = logging.getLogger(__name__)


class ImageDescriptionAnalyzer:
    """
    Generates human-readable descriptions of images using Gemma4 LLM.

    Communicates with Ollama API running Gemma4 model to analyze image content.
    Images are encoded to base64 and sent to the model for analysis.

    Attributes:
        model_version: Semantic version string of the underlying ML model.
        gemma_api_url: Base URL to the Gemma/Ollama API endpoint.
        model_name: Name of the Ollama model to use.
    """

    model_version: str = "0.1.0"

    def __init__(self) -> None:
        """Initialize the image description analyzer with Gemma API configuration."""
        self.gemma_api_url = os.getenv("GEMMA_API", "http://gemma:11434")
        self.model_name = os.getenv("MODEL_NAME", "gemma4:e2b")

        logger.debug(
            "ImageDescriptionAnalyzer initialized: api=%s model=%s",
            self.gemma_api_url,
            self.model_name,
        )

    def describe_and_arbitrate(
        self,
        image_paths: list[str],
        vit_label: str | None = None,
        streetclip_results: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a description and arbitrate location using Gemma4.
        """
        if not image_paths:
            logger.warning("ImageDescriptionAnalyzer: no image paths provided")
            return {
                "description": "",
                "confidence": 0.0,
                "objects_identified": [],
                "scenes": [],
                "arbitrated_location": None,
                "metadata": {"model_used": self.model_name, "processing_time_ms": 0, "error": "No image paths"},
            }

        image_path = image_paths[0]

        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            vit_text = f"ViT: {vit_label}" if vit_label else "ViT: Brak"
            sc_list = [h["label"] for h in (streetclip_results or [])[:3]]
            sc_text = f"StreetCLIP: {', '.join(sc_list)}" if sc_list else "StreetCLIP: Brak"

            prompt = (
                "Jesteś zaawansowanym asystentem AI. Przeanalizuj to zdjęcie i zwróć wynik WYŁĄCZNIE w formacie JSON.\n"
                "Mamy podpowiedzi z innych modeli AI dotyczące lokalizacji na zdjęciu:\n"
                f"- {vit_text}\n"
                f"- {sc_text}\n\n"
                "Zwróć JSON o strukturze:\n"
                "{\n"
                '  "description": "Zwięzły, jednozdaniowy opis tego co widzisz na zdjęciu.",\n'
                '  "objects_identified": ["obiekt1", "obiekt2"],\n'
                '  "scenes": ["scena1", "scena2"],\n'
                '  "arbitrated_location": "Dokładna nazwa lokalizacji (zabytek, miasto, kraj) np. Pałac Kultury i Nauki, Warszawa, Polska oparta na podpowiedziach i obrazie. Jeśli żadna podpowiedź nie pasuje, wpisz UNKNOWN."\n'
                "}\n"
                "Odpowiadaj WYŁĄCZNIE poprawnym obiektem JSON, żadnych komentarzy, żadnych wcięć poza JSONem."
            )

            response = requests.post(
                f"{self.gemma_api_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False,
                },
                timeout=300,
            )
            response.raise_for_status()
            response_data = response.json()
            generated_text = response_data.get("response", "").strip()

            import json
            import re
            parsed = {}
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    logger.warning("ImageDescriptionAnalyzer: JSON decode failed")

            result = {
                "description": parsed.get("description", "Nie udało się wygenerować opisu."),
                "confidence": 0.85,
                "objects_identified": parsed.get("objects_identified", []),
                "scenes": parsed.get("scenes", []),
                "arbitrated_location": parsed.get("arbitrated_location"),
                "metadata": {
                    "model_used": self.model_name,
                    "processing_time_ms": response_data.get("eval_duration", 0) // 1_000_000,
                },
            }
            return result

        except Exception as e:
            logger.exception("ImageDescriptionAnalyzer: unexpected error: %s", e)
            return {
                "description": "",
                "confidence": 0.0,
                "objects_identified": [],
                "scenes": [],
                "arbitrated_location": None,
                "metadata": {"model_used": self.model_name, "processing_time_ms": 0, "error": str(e)},
            }
