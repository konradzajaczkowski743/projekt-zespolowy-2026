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

    def describe_image(self, image_paths: list[str]) -> dict[str, Any]:
        """
        Generate a detailed description of the image content using Gemma4.

        This method reads the first image from the provided paths, encodes it
        to base64, sends it to the Gemma4 model via Ollama API, and returns
        a structured description.

        Args:
            image_paths: List of file paths to images to describe. Only the
                first image is processed.

        Returns:
            Dictionary with structure::

                {
                    "description": "<detailed description of image content>",
                    "confidence": <float 0.0-1.0>,
                    "objects_identified": [<list of identified objects>],
                    "scenes": [<list of identified scenes>],
                    "metadata": {
                        "model_used": "<model name>",
                        "processing_time_ms": <milliseconds>
                    }
                }

        Raises:
            FileNotFoundError: If the image file cannot be found.
            requests.RequestException: If communication with Gemma API fails.
        """
        if not image_paths:
            logger.warning("ImageDescriptionAnalyzer.describe_image: no image paths provided")
            return {
                "description": "",
                "confidence": 0.0,
                "objects_identified": [],
                "scenes": [],
                "metadata": {
                    "model_used": self.model_name,
                    "processing_time_ms": 0,
                    "error": "No image paths provided",
                },
            }

        image_path = image_paths[0]

        try:
            logger.debug(
                "ImageDescriptionAnalyzer.describe_image: reading image=%s",
                image_path,
            )

            # Read and encode image to base64
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Prepare prompt for Gemma
            prompt = (
                "Wciel się w rolę obserwatora. Opisz jednym naturalnym, płynnym zdaniem "
                "co widzisz na zdjęciu i gdzie myślisz, że jesteś. Na przykład: "
                "'Widzę wieżę Eiffla, dużo osób i drzewa, więc myślę, że jestem w Paryżu we Francji.' "
                "albo 'Nie wiem dokładnie gdzie jestem, ale rozpoznaję plażę i roślinność "
                "typową dla Europy Południowej.' Nie używaj punktatorów, po prostu napisz "
                "to jako swoje przemyślenie."
            )

            # Call Gemma API via Ollama
            logger.debug(
                "ImageDescriptionAnalyzer: calling Gemma API at %s",
                self.gemma_api_url,
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

            logger.info(
                "ImageDescriptionAnalyzer: successfully generated description "
                "for image=%s model=%s",
                image_path,
                self.model_name,
            )

            # Parse the response to extract structured information
            result = {
                "description": generated_text,
                "confidence": 0.85,  # Gemma confidence is moderate
                "objects_identified": self._extract_objects(generated_text),
                "scenes": self._extract_scenes(generated_text),
                "metadata": {
                    "model_used": self.model_name,
                    "processing_time_ms": response_data.get("eval_duration", 0) // 1_000_000,
                },
            }

            logger.debug(
                "ImageDescriptionAnalyzer.describe_image result: %s", result
            )
            return result

        except FileNotFoundError as e:
            logger.error("ImageDescriptionAnalyzer: file not found: %s", e)
            return {
                "description": "",
                "confidence": 0.0,
                "objects_identified": [],
                "scenes": [],
                "metadata": {
                    "model_used": self.model_name,
                    "processing_time_ms": 0,
                    "error": str(e),
                },
            }
        except requests.RequestException as e:
            logger.error(
                "ImageDescriptionAnalyzer: Gemma API request failed: %s", e
            )
            return {
                "description": "",
                "confidence": 0.0,
                "objects_identified": [],
                "scenes": [],
                "metadata": {
                    "model_used": self.model_name,
                    "processing_time_ms": 0,
                    "error": f"Gemma API error: {str(e)}",
                },
            }
        except Exception as e:
            logger.exception(
                "ImageDescriptionAnalyzer.describe_image: unexpected error: %s", e
            )
            return {
                "description": "",
                "confidence": 0.0,
                "objects_identified": [],
                "scenes": [],
                "metadata": {
                    "model_used": self.model_name,
                    "processing_time_ms": 0,
                    "error": f"Unexpected error: {str(e)}",
                },
            }

    def _extract_objects(self, text: str) -> list[str]:
        """
        Extract object names from Gemma's response text.

        Simple heuristic extraction - looks for common patterns in the response.

        Args:
            text: The generated description text from Gemma.

        Returns:
            List of identified objects.
        """
        # This is a simple extraction; in production you might use regex or NLP
        objects = []
        keywords = [
            "osób", "osoba", "człowiek", "człowieka",
            "samochód", "samochodu", "auto",
            "pies", "psa", "kot", "kota",
            "drzewo", "drzewa", "roślina", "roślin",
            "budynek", "budynku", "dom", "domów",
            "ulica", "droga", "ścieżka",
            "niebo", "chmury", "słońce",
        ]

        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                objects.append(keyword)

        return list(set(objects))  # Remove duplicates

    def _extract_scenes(self, text: str) -> list[str]:
        """
        Extract scene types from Gemma's response text.

        Simple heuristic extraction - looks for common scene patterns.

        Args:
            text: The generated description text from Gemma.

        Returns:
            List of identified scene types.
        """
        scenes = []
        scene_keywords = [
            "wewnątrz", "exterior", "zewnątrz", "na dworze",
            "dzień", "noc", "zachód", "wschód",
            "park", "plaża", "las", "pole", "miasto",
            "biuro", "szkoła", "szpital", "sklep",
            "natura", "krajobraz", "pejzaż",
        ]

        text_lower = text.lower()
        for keyword in scene_keywords:
            if keyword in text_lower:
                scenes.append(keyword)

        return list(set(scenes))  # Remove duplicates
