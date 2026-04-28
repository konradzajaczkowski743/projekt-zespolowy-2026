"""
geolocation/loaders.py

Singleton cache dla modeli ViT używanych w module geolokalizacji.
Wzorzec identyczny jak w pozostałych modułach — każdy Celery worker
ładuje wagi dokładnie raz przy pierwszym wywołaniu.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers.models.vit import ViTForImageClassification, ViTImageProcessor

_lock = threading.Lock()
_cache: dict[str, "_ModelBundle"] = {}


@dataclass
class _ModelBundle:
    processor: ViTImageProcessor
    model: ViTForImageClassification
    labels: list[str] = field(default_factory=list)


def _load(model_path: str) -> _ModelBundle:
    """Ładuje model z lokalnego katalogu (local_files_only=True)."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model nie istnieje pod ścieżką: {path}. "
            "Sprawdź, czy etap 'model-downloader' w Dockerfile zakończył się poprawnie."
        )

    processor = ViTImageProcessor.from_pretrained(
        str(path),
        local_files_only=True,
    )
    model = ViTForImageClassification.from_pretrained(
        str(path),
        local_files_only=True,
        torch_dtype=torch.float32,  # CPU-safe; zmień na float16 dla GPU
    )
    model.eval()

    labels: list[str] = []
    if hasattr(model.config, "id2label"):
        labels = [model.config.id2label[i] for i in sorted(model.config.id2label)]

    return _ModelBundle(processor=processor, model=model, labels=labels)


def get_landmarks_model() -> _ModelBundle:
    """Zwraca (tworząc raz) model mmgyorke/vit-world-landmarks."""
    key = "landmarks"
    if key not in _cache:
        with _lock:
            if key not in _cache:
                path = os.environ["MODEL_LANDMARKS_PATH"]
                _cache[key] = _load(path)
    return _cache[key]


def get_places365_model() -> _ModelBundle:
    """Zwraca (tworząc raz) model corenet-community/places365-224x224-vit-base."""
    key = "places365"
    if key not in _cache:
        with _lock:
            if key not in _cache:
                path = os.environ["MODEL_PLACES365_PATH"]
                _cache[key] = _load(path)
    return _cache[key]