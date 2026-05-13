"""
geolocation/geo_classifier.py

Klasyfikacja zdjęcia przy użyciu obu modeli ViT:
  - mmgyorke/vit-world-landmarks  → rozpoznawanie konkretnych zabytków
  - corenet-community/places365   → klasyfikacja typu sceny (365 kategorii)

Oba modele przyjmują PIL.Image, zwracają top-k etykiet z pewnością.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

from .loaders import get_landmarks_model, get_places365_model

TOP_K = 5


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    score: float


@dataclass
class GeoClassificationOutput:
    landmarks: list[ClassificationResult]   # z vit-world-landmarks
    scene_types: list[ClassificationResult] # z places365
    top_landmark: str | None                # najlepszy zabytek (score > threshold)
    top_scene: str | None                   # najlepsza scena
    landmark_confidence: float
    scene_confidence: float


_LANDMARK_THRESHOLD = 0.35  # poniżej tego — nie uznajemy zabytkowego dopasowania
_SCENE_THRESHOLD = 0.20


def _run_vit(image: Image.Image, model_key: str) -> list[ClassificationResult]:
    """Wspólna logika inferencji dla obu modeli ViT."""
    loader = get_landmarks_model if model_key == "landmarks" else get_places365_model
    bundle = loader()

    inputs = bundle.processor(images=image, return_tensors="pt")

    with torch.inference_mode():
        logits = bundle.model(**inputs).logits  # (1, num_classes)

    probs = torch.softmax(logits, dim=-1).squeeze(0)
    top_scores, top_indices = probs.topk(min(TOP_K, probs.size(0)))

    results: list[ClassificationResult] = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        label = bundle.labels[idx] if bundle.labels else str(idx)
        results.append(ClassificationResult(label=label, score=round(score, 4)))

    return results


def classify_image(image: Image.Image) -> GeoClassificationOutput:
    """
    Uruchamia oba modele na tym samym zdjęciu.
    Wywoływany z geolocation/tasks.py wewnątrz Celery taska.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    landmarks = _run_vit(image, "landmarks")
    scene_types = _run_vit(image, "places365")

    top_landmark_hit = landmarks[0] if landmarks else None
    top_scene_hit = scene_types[0] if scene_types else None

    return GeoClassificationOutput(
        landmarks=landmarks,
        scene_types=scene_types,
        top_landmark=(
            top_landmark_hit.label
            if top_landmark_hit and top_landmark_hit.score >= _LANDMARK_THRESHOLD
            else None
        ),
        top_scene=(
            top_scene_hit.label
            if top_scene_hit and top_scene_hit.score >= _SCENE_THRESHOLD
            else None
        ),
        landmark_confidence=top_landmark_hit.score if top_landmark_hit else 0.0,
        scene_confidence=top_scene_hit.score if top_scene_hit else 0.0,
    )