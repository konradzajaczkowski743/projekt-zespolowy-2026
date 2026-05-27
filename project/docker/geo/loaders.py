"""
geolocation/loaders.py

Singleton cache for ViT and CLIP models used in the geolocation module.
Each Celery worker loads model weights exactly once on first invocation.

Models:
    - mmgyorke/vit-world-landmarks  → landmark recognition
    - corenet-community/places365   → scene classification
    - geolocal/StreetCLIP           → zero-shot geolocation
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers.models.vit import ViTForImageClassification, ViTImageProcessor

_lock = threading.Lock()
_cache: dict[str, "_ModelBundle | _CLIPBundle"] = {}


@dataclass
class _ModelBundle:
    processor: ViTImageProcessor
    model: ViTForImageClassification
    labels: list[str] = field(default_factory=list)


@dataclass
class _CLIPBundle:
    processor: object  # CLIPProcessor
    model: object      # CLIPModel
    labels: list[str] = field(default_factory=list)


def _load_vit(model_path: str) -> _ModelBundle:
    """Load a ViT model from a local directory (local_files_only=True)."""
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
        torch_dtype=torch.float32,
    )
    model.eval()

    labels: list[str] = []
    if hasattr(model.config, "id2label"):
        labels = [model.config.id2label[i] for i in sorted(model.config.id2label)]

    return _ModelBundle(processor=processor, model=model, labels=labels)


def _load_clip(model_path: str) -> _CLIPBundle:
    """Load a CLIP model from a local directory (local_files_only=True)."""
    from transformers import CLIPModel, CLIPProcessor

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model nie istnieje pod ścieżką: {path}. "
            "Sprawdź, czy etap 'model-downloader' w Dockerfile zakończył się poprawnie."
        )

    processor = CLIPProcessor.from_pretrained(
        str(path),
        local_files_only=True,
    )
    model = CLIPModel.from_pretrained(
        str(path),
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.eval()

    return _CLIPBundle(processor=processor, model=model)


def get_landmarks_model() -> _ModelBundle:
    """Return (create once) the mmgyorke/vit-world-landmarks model."""
    key = "landmarks"
    if key not in _cache:
        with _lock:
            if key not in _cache:
                path = os.environ["MODEL_LANDMARKS_PATH"]
                _cache[key] = _load_vit(path)
    return _cache[key]


def get_places365_model() -> _ModelBundle:
    """Return (create once) the corenet-community/places365-224x224-vit-base model."""
    key = "places365"
    if key not in _cache:
        with _lock:
            if key not in _cache:
                path = os.environ["MODEL_PLACES365_PATH"]
                _cache[key] = _load_vit(path)
    return _cache[key]


def get_streetclip_model() -> _CLIPBundle:
    """Return (create once) the geolocal/StreetCLIP model."""
    key = "streetclip"
    if key not in _cache:
        with _lock:
            if key not in _cache:
                path = os.environ["MODEL_STREETCLIP_PATH"]
                _cache[key] = _load_clip(path)
    return _cache[key]