"""
geolocation/geo_classifier.py

Classification of images using three models:
  - mmgyorke/vit-world-landmarks  → specific landmark recognition
  - corenet-community/places365   → scene type classification (365 categories)
  - geolocal/StreetCLIP           → zero-shot geolocation with candidate labels

All models accept PIL.Image and return top-k labels with confidence scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

from .loaders import get_landmarks_model, get_places365_model, get_streetclip_model

TOP_K = 5


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    score: float


@dataclass
class GeoClassificationOutput:
    landmarks: list[ClassificationResult]        # from vit-world-landmarks
    scene_types: list[ClassificationResult]       # from places365
    streetclip_predictions: list[ClassificationResult]  # from StreetCLIP
    top_landmark: str | None                      # best landmark (score > threshold)
    top_scene: str | None                         # best scene
    top_streetclip: str | None                    # best StreetCLIP prediction
    landmark_confidence: float
    scene_confidence: float
    streetclip_confidence: float


_LANDMARK_THRESHOLD = 0.35
_SCENE_THRESHOLD = 0.20
_STREETCLIP_THRESHOLD = 0.10

# Full candidate labels for StreetCLIP zero-shot classification
POLISH_CITIES: list[str] = [
    "Warszawa, Polska", "Kraków, Polska", "Wrocław, Polska", "Łódź, Polska",
    "Poznań, Polska", "Gdańsk, Polska", "Szczecin, Polska", "Bydgoszcz, Polska",
    "Lublin, Polska", "Białystok, Polska", "Katowice, Polska", "Gdynia, Polska",
    "Częstochowa, Polska", "Radom, Polska", "Toruń, Polska", "Sosnowiec, Polska",
    "Kielce, Polska", "Rzeszów, Polska", "Gliwice, Polska", "Zabrze, Polska",
    "Olsztyn, Polska", "Bielsko-Biała, Polska", "Bytom, Polska",
    "Zielona Góra, Polska", "Rybnik, Polska", "Ruda Śląska, Polska",
    "Opole, Polska", "Tychy, Polska", "Gorzów Wielkopolski, Polska",
    "Elbląg, Polska", "Płock, Polska", "Dąbrowa Górnicza, Polska",
    "Wałbrzych, Polska", "Włocławek, Polska", "Tarnów, Polska",
    "Chorzów, Polska", "Koszalin, Polska", "Kalisz, Polska", "Legnica, Polska",
    "Grudziądz, Polska", "Jaworzno, Polska", "Słupsk, Polska",
    "Jastrzębie-Zdrój, Polska", "Nowy Sącz, Polska", "Jelenia Góra, Polska",
    "Siedlce, Polska", "Mysłowice, Polska", "Konin, Polska", "Piła, Polska",
]

EUROPEAN_CITIES: list[str] = [
    "London, United Kingdom", "Paris, France", "Berlin, Germany",
    "Madrid, Spain", "Rome, Italy", "Amsterdam, Netherlands",
    "Vienna, Austria", "Prague, Czech Republic", "Budapest, Hungary",
    "Barcelona, Spain", "Munich, Germany", "Milan, Italy",
    "Lisbon, Portugal", "Brussels, Belgium", "Stockholm, Sweden",
    "Oslo, Norway", "Copenhagen, Denmark", "Helsinki, Finland",
    "Dublin, Ireland", "Zurich, Switzerland", "Athens, Greece",
    "Istanbul, Turkey", "Bucharest, Romania", "Sofia, Bulgaria",
    "Zagreb, Croatia", "Belgrade, Serbia", "Bratislava, Slovakia",
    "Ljubljana, Slovenia", "Tallinn, Estonia", "Riga, Latvia",
    "Vilnius, Lithuania", "Kyiv, Ukraine",
    "Moscow, Russia", "Saint Petersburg, Russia",
    "Edinburgh, Scotland", "Hamburg, Germany", "Frankfurt, Germany",
    "Lyon, France", "Marseille, France", "Naples, Italy",
    "Florence, Italy", "Venice, Italy", "Seville, Spain",
    "Porto, Portugal", "Gothenburg, Sweden", "Antwerp, Belgium",
    "Dresden, Germany", "Salzburg, Austria", "Dubrovnik, Croatia",
]

POLISH_LANDMARKS: list[str] = [
    "Pałac Kultury i Nauki, Warszawa",
    "Zamek Królewski, Warszawa",
    "Stare Miasto, Warszawa",
    "Wawel Castle, Kraków",
    "Sukiennice, Kraków",
    "Kościół Mariacki, Kraków",
    "Kazimierz, Kraków",
    "Hala Stulecia, Wrocław",
    "Ostrów Tumski, Wrocław",
    "Stary Rynek, Poznań",
    "Długi Targ, Gdańsk",
    "Żuraw Gdański, Gdańsk",
    "Malbork Castle, Malbork",
    "Zamość Old Town, Zamość",
    "Wieliczka Salt Mine, Wieliczka",
    "Jasna Góra, Częstochowa",
    "Łazienki Park, Warszawa",
    "Wilanów Palace, Warszawa",
    "Centennial Hall, Wrocław",
    "Manufaktura, Łódź",
]

ALL_CANDIDATE_LABELS: list[str] = POLISH_LANDMARKS + POLISH_CITIES + EUROPEAN_CITIES


def _run_vit(image: Image.Image, model_key: str) -> list[ClassificationResult]:
    """Common inference logic for both ViT models."""
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


def _run_streetclip(
    image: Image.Image,
    candidate_labels: list[str] | None = None,
) -> list[ClassificationResult]:
    """Zero-shot classification using StreetCLIP."""
    bundle = get_streetclip_model()
    labels = candidate_labels or ALL_CANDIDATE_LABELS

    inputs = bundle.processor(
        text=labels, images=image, return_tensors="pt", padding=True
    )

    with torch.inference_mode():
        outputs = bundle.model(**inputs)
        logits = outputs.logits_per_image  # (1, num_labels)
        probs = logits.softmax(dim=-1).squeeze(0)

    top_scores, top_indices = probs.topk(min(TOP_K, len(labels)))

    results: list[ClassificationResult] = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        results.append(ClassificationResult(
            label=labels[idx], score=round(score, 4)
        ))

    return results


def classify_image(image: Image.Image) -> GeoClassificationOutput:
    """
    Run all three models on the same image.

    Called from geolocation tasks inside a Celery task.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    landmarks = _run_vit(image, "landmarks")
    scene_types = _run_vit(image, "places365")
    streetclip_preds = _run_streetclip(image)

    top_landmark_hit = landmarks[0] if landmarks else None
    top_scene_hit = scene_types[0] if scene_types else None
    top_streetclip_hit = streetclip_preds[0] if streetclip_preds else None

    return GeoClassificationOutput(
        landmarks=landmarks,
        scene_types=scene_types,
        streetclip_predictions=streetclip_preds,
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
        top_streetclip=(
            top_streetclip_hit.label
            if top_streetclip_hit and top_streetclip_hit.score >= _STREETCLIP_THRESHOLD
            else None
        ),
        landmark_confidence=top_landmark_hit.score if top_landmark_hit else 0.0,
        scene_confidence=top_scene_hit.score if top_scene_hit else 0.0,
        streetclip_confidence=top_streetclip_hit.score if top_streetclip_hit else 0.0,
    )