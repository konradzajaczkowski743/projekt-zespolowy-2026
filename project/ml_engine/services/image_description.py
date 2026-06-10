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
import time
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

    def query_image_with_context(
        self,
        image_path: str,
        prompt: str,
    ) -> str:
        """
        Answer a user query about an image with analysis context.

        Sends the image and a detailed prompt to Gemma4 to answer
        specific questions about the analyzed image.

        Args:
            image_path: Path to the image file.
            prompt: The user's question or custom prompt.

        Returns:
            The AI-generated response to the query.
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

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

            return generated_text

        except Exception as e:
            logger.exception("ImageDescriptionAnalyzer.query_image_with_context: error: %s", e)
            return f"Błąd przy przetwarzaniu pytania: {str(e)}"

    def query_with_analysis_context(
        self,
        analysis_result: dict[str, Any],
        user_prompt: str,
        max_retries: int = 3,
    ) -> str:
        """
        Answer a user query using ONLY previous analysis results (no image).

        This method is optimized for follow-up queries after the initial analysis.
        It works with the textual analysis results to avoid reprocessing the image
        and causing OOM errors in Gemma.

        Args:
            analysis_result: Dictionary from describe_and_arbitrate() containing:
                - description: Image description
                - objects_identified: List of detected objects
                - scenes: List of detected scenes
                - arbitrated_location: Identified location
            user_prompt: The user's follow-up question.
            max_retries: Number of retry attempts on server errors.

        Returns:
            The AI-generated response to the query.
        """
        try:
            # Build context from previous analysis
            description = analysis_result.get("description", "Brak opisu")
            objects = ", ".join(analysis_result.get("objects_identified", []))
            scenes = ", ".join(analysis_result.get("scenes", []))
            location = analysis_result.get("arbitrated_location", "Nieznana")
            detected_objects = analysis_result.get("objects_detected", []) or []
            alpr_results = analysis_result.get("alpr", []) or []

            if detected_objects:
                label_counts: dict[str, int] = {}
                for detected in detected_objects:
                    label = detected.get("label", "unknown")
                    label_counts[label] = label_counts.get(label, 0) + 1
                count_lines = "\n".join(
                    f"- {label}: {count}" for label, count in sorted(label_counts.items())
                )
                object_detection_summary = (
                    f"Detected objects: {len(detected_objects)}\n"
                    f"{count_lines}\n"
                )
            else:
                object_detection_summary = "Detected objects: brak\n"

            context_prompt = (
                "Na podstawie poprzedniej analizy zdjęcia, odpowiedz na pytanie użytkownika.\n\n"
                "=== POPRZEDNIA ANALIZA ===\n"
                f"Opis: {description}\n"
                f"Obiekty: {objects if objects else 'brak'}\n"
                f"Sceny: {scenes if scenes else 'brak'}\n"
                f"Lokalizacja: {location}\n"
                "=== DETEKTOWANE OBIEKTY ===\n"
                f"{object_detection_summary}\n"
                "=== ALPR (TABLICE REJESTRACYJNE) ===\n"
                f"{(', '.join([a.get('plate_text','') for a in alpr_results]) if alpr_results else 'brak')}\n"
                "=== PYTANIE UŻYTKOWNIKA ===\n"
                f"{user_prompt}\n\n"
                "Odpowiedz zwięźle i precyzyjnie, bazując na powyższej analizie.\n"
                "Jeżeli pytanie dotyczy liczby obiektów, podaj dokładną liczbę na podstawie wykrytych obiektów.\n"
                "Jeżeli pytanie dotyczy tablic rejestracyjnych, odnieś się do sekcji ALPR powyżej.\n"
                "Nie dodawaj informacji wykraczających poza te dane i nie sugeruj się domniemaną wielokrotnością."
            )

            logger.debug(f"Querying Gemma with analysis context (no image)")

            # Retry logic for temporary failures
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{self.gemma_api_url}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": context_prompt,
                            "stream": False,
                        },
                        timeout=300,
                    )

                    if response.status_code == 500:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries}: Gemma returned 500, retrying..."
                        )
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        else:
                            return "Błąd serwera Gemma. Spróbuj ponownie za chwilę."

                    response.raise_for_status()
                    response_data = response.json()
                    generated_text = response_data.get("response", "").strip()

                    logger.debug("Gemma query completed successfully")
                    return generated_text

                except requests.exceptions.Timeout:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries}: Timeout, retrying..."
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        return "Timeout. Spróbuj ponownie za chwilę."

                except requests.exceptions.ConnectionError as e:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries}: Connection error: {e}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        return f"Błąd połączenia: {str(e)}"

            return "Nie udało się połączyć z Gemmą po kilku próbach."

        except Exception as e:
            logger.exception(
                "ImageDescriptionAnalyzer.query_with_analysis_context: error: %s", e
            )
            return f"Błąd przy przetwarzaniu pytania: {str(e)}"
