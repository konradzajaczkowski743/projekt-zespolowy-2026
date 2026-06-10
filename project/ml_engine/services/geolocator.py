"""
Geo-verification analysis service for the project.

This module provides ``GeoVerificationAnalyzer`` for determining the geographic
location depicted in an image **based solely on visual content**. Supplied GPS
co-ordinates (from EXIF or user input) are used only as a secondary comparison
signal — they never influence the visual prediction itself.

Architecture — three-layer visual recognition:

1. **ViT World Landmarks** (``mmgyorke/vit-world-landmarks``) — fast classifier
   for well-known world landmarks.
2. **StreetCLIP** (``geolocal/StreetCLIP``) — zero-shot CLIP-based geolocation
   with configurable candidate labels (Polish cities, European capitals, etc.).
3. **Gemma4 VLM** (via Ollama) — multimodal LLM description with location
   extraction as a final fallback.

Geo-coding is performed locally via the **Photon** container (already in
docker-compose) — no external API calls are made.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.parse
import urllib.request
from typing import Any

from PIL import Image

logger: logging.Logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("[geolocator] PyTorch not installed — ML models disabled")

CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", "/tmp/models")
_models_cache: dict[str, Any] = {}

# ── Photon local geocoder ────────────────────────────────────────────────────

PHOTON_HOST = os.getenv("PHOTON_HOST", "photon")
PHOTON_PORT = os.getenv("PHOTON_PORT", "2322")


def _geocode_via_photon(location_name: str) -> tuple[float, float] | None:
    """
    Geocode a location name using the local Photon container.

    Falls back to ``None`` if Photon is unreachable or returns no results.
    No external API is ever called.
    """
    try:
        clean_name = (
            location_name
            .replace("Landmark: ", "")
            .replace("Scene: ", "")
            .replace("StreetCLIP: ", "")
            .replace("Gemma: ", "")
            .strip()
        )
        url = (
            f"http://{PHOTON_HOST}:{PHOTON_PORT}/api"
            f"?q={urllib.parse.quote(clean_name)}&limit=1"
        )
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            features = data.get("features", [])
            if features:
                coords = features[0]["geometry"]["coordinates"]
                return float(coords[1]), float(coords[0])  # lat, lon
    except Exception as e:
        logger.warning("[geolocator] Photon geocoding failed for '%s': %s", location_name, e)
    return None


# ── Haversine distance ───────────────────────────────────────────────────────


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in kilometres."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ── Device helper ────────────────────────────────────────────────────────────


def _get_device() -> Any:
    if not HAS_TORCH:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Model loading (ViT) ─────────────────────────────────────────────────────


def _load_vit_model(repo_id: str, cache_key: str) -> tuple[Any, Any]:
    """Load a ViT classification model with singleton cache."""
    if cache_key not in _models_cache:
        if not HAS_TORCH:
            return None, None

        from transformers import ViTForImageClassification, ViTImageProcessor

        logger.info("[geolocator] Loading %s...", repo_id)
        try:
            processor = ViTImageProcessor.from_pretrained(repo_id, cache_dir=CACHE_DIR)
            model = ViTForImageClassification.from_pretrained(repo_id, cache_dir=CACHE_DIR)
            device = _get_device()
            if device is not None:
                model = model.to(device)
            _models_cache[f"{cache_key}_processor"] = processor
            _models_cache[cache_key] = model.eval()
            logger.info("[geolocator] %s loaded.", repo_id)
        except Exception as exc:
            logger.exception("[geolocator] Error loading %s: %s", repo_id, exc)
            _models_cache[cache_key] = None
            _models_cache[f"{cache_key}_processor"] = None

    return _models_cache.get(f"{cache_key}_processor"), _models_cache.get(cache_key)


# ── StreetCLIP loading & inference ───────────────────────────────────────────

# Full label set: Polish cities (>100k population) + major European cities
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
    "Vilnius, Lithuania", "Kyiv, Ukraine", "Minsk, Belarus",
    "Moscow, Russia", "Saint Petersburg, Russia",
    "Edinburgh, Scotland", "Hamburg, Germany", "Frankfurt, Germany",
    "Lyon, France", "Marseille, France", "Naples, Italy",
    "Florence, Italy", "Venice, Italy", "Seville, Spain",
    "Porto, Portugal", "Gothenburg, Sweden", "Antwerp, Belgium",
    "Dresden, Germany", "Salzburg, Austria", "Dubrovnik, Croatia",
]

# Polish landmark names to help StreetCLIP with well-known places
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

# Load dynamic landmarks from our local JSON DB
try:
    _db_path = os.path.join(os.path.dirname(__file__), "..", "data", "polish_landmarks_db.json")
    with open(_db_path, "r", encoding="utf-8") as f:
        _local_db = json.load(f)
        for _lm in _local_db:
            if _lm.get("name") and _lm["name"] not in POLISH_LANDMARKS:
                POLISH_LANDMARKS.append(_lm["name"])
    logger.info("[geolocator] Loaded %d landmarks from local DB.", len(POLISH_LANDMARKS))
except Exception as e:
    logger.warning("[geolocator] Failed to load polish_landmarks_db.json: %s", e)

# Load vector database
_vector_db = {}
try:
    if HAS_TORCH:
        import torch
        _vec_path = os.path.join(os.path.dirname(__file__), "..", "data", "landmark_vectors.pt")
        if os.path.exists(_vec_path):
            _vector_db = torch.load(_vec_path, map_location="cpu", weights_only=True)
            logger.info("[geolocator] Loaded %d vectors from local vector DB.", len(_vector_db))
except Exception as e:
    logger.warning("[geolocator] Failed to load landmark_vectors.pt: %s", e)

ALL_CANDIDATE_LABELS: list[str] = POLISH_LANDMARKS + POLISH_CITIES + EUROPEAN_CITIES


def _load_streetclip() -> tuple[Any, Any] | None:
    """Load StreetCLIP model with singleton cache."""
    cache_key = "streetclip"
    if cache_key not in _models_cache:
        if not HAS_TORCH:
            return None

        try:
            from transformers import CLIPModel, CLIPProcessor

            repo_id = "geolocal/StreetCLIP"
            logger.info("[geolocator] Loading StreetCLIP...")
            processor = CLIPProcessor.from_pretrained(repo_id, cache_dir=CACHE_DIR)
            model = CLIPModel.from_pretrained(repo_id, cache_dir=CACHE_DIR)
            device = _get_device()
            if device is not None:
                model = model.to(device)
            _models_cache[f"{cache_key}_processor"] = processor
            _models_cache[cache_key] = model.eval()
            logger.info("[geolocator] StreetCLIP loaded.")
        except Exception as exc:
            logger.exception("[geolocator] Error loading StreetCLIP: %s", exc)
            _models_cache[cache_key] = None
            _models_cache[f"{cache_key}_processor"] = None

    proc = _models_cache.get(f"{cache_key}_processor")
    mdl = _models_cache.get(cache_key)
    if proc is None or mdl is None:
        return None
    return proc, mdl


def _classify_with_streetclip(
    image: Image.Image,
    candidate_labels: list[str] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Zero-shot location classification using StreetCLIP.

    Returns top-k candidate locations with probabilities, sorted descending.
    """
    if not HAS_TORCH:
        return []

    bundle = _load_streetclip()
    if bundle is None:
        return []

    processor, model = bundle
    labels = candidate_labels or ALL_CANDIDATE_LABELS
    device = _get_device()

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image  # (1, num_labels)
        probs = logits.softmax(dim=-1).squeeze(0)

    top_scores, top_indices = probs.topk(min(top_k, len(labels)))

    results: list[dict[str, Any]] = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        results.append({
            "label": labels[idx],
            "score": round(score, 4),
        })

    return results





# ── Main analyser class ─────────────────────────────────────────────────────


class GeoVerificationAnalyzer:
    """
    Determines geographic location from image content using visual analysis.

    The analyzer uses a 3-layer approach:
    1. ViT landmarks — fast identification of well-known world landmarks
    2. StreetCLIP — zero-shot geolocation with Polish/European candidate labels
    3. Gemma4 VLM — multimodal description-based location extraction

    GPS coordinates from EXIF/user input are used **only for comparison** —
    they never influence the visual prediction. This ensures the same image
    always produces the same predicted location regardless of metadata.

    Attributes:
        model_version: Semantic version string of the underlying ML models.
    """

    model_version: str = "2.0.0"

    def __init__(self) -> None:
        """Initialise the analyser."""
        logger.debug(
            "GeoVerificationAnalyzer initialised (version=%s)", self.model_version
        )

    def verify_location(
        self,
        image_paths: list[str],
        latitude: float | None,
        longitude: float | None,
    ) -> dict[str, Any]:
        """
        Determine the location depicted in an image and optionally compare
        with supplied GPS coordinates.

        The visual prediction is **always independent** of supplied coordinates.
        Coordinates serve only as a secondary comparison signal.

        Args:
            image_paths: Paths to image files to analyse.
            latitude: GPS latitude from EXIF/user (comparison only).
            longitude: GPS longitude from EXIF/user (comparison only).

        Returns:
            Dictionary with keys:
                - predicted_region: visually predicted location name
                - prediction_source: which layer made the prediction
                - is_location_consistent: True if visual matches GPS (if GPS given)
                - confidence_score: prediction confidence
                - distance_km: distance between visual and GPS location (if both available)
                - streetclip_top5: top-5 StreetCLIP candidates
                - description_location: location extracted from Gemma description
        """
        logger.info(
            "GeoVerificationAnalyzer.verify_location: images=%d lat=%s lon=%s",
            len(image_paths),
            latitude,
            longitude,
        )

        # ── Layer 1-3: Visual prediction (independent of coordinates!) ───────
        prediction = self._predict_region_multilayer(image_paths)

        predicted_region: str | None = prediction.get("predicted_region")
        prediction_source: str = prediction.get("source", "none")
        visual_confidence: float = prediction.get("confidence", 0.0)
        streetclip_top5: list[dict] = prediction.get("streetclip_top5", [])
        description_location: str | None = prediction.get("description_location")

        # ── Comparison with GPS (if provided) ────────────────────────────────
        is_consistent: bool = True
        distance_km: float | None = None

        if latitude is not None and longitude is not None and predicted_region:
            is_consistent, distance_km = self.calculate_distance(predicted_region, latitude, longitude)

        result: dict[str, Any] = {
            "is_location_consistent": is_consistent,
            "confidence_score": round(visual_confidence, 4),
            "predicted_region": predicted_region,
            "vit_predicted_region": prediction.get("vit_predicted_region"),
            "prediction_source": prediction_source,
            "distance_km": distance_km,
            "streetclip_top5": streetclip_top5,
            "description_location": description_location,
        }
        logger.debug("GeoVerificationAnalyzer.verify_location result: %s", result)
        return result

    def _predict_region_multilayer(
        self,
        image_paths: list[str],
    ) -> dict[str, Any]:
        """
        3-layer visual location prediction — runs layers sequentially,
        returning as soon as a confident prediction is found.
        """
        if not image_paths:
            return {"predicted_region": None, "source": "none", "confidence": 0.0}

        image_path = image_paths[0]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error("[geolocator] Error reading image: %s", e)
            return {"predicted_region": None, "source": "error", "confidence": 0.0}

        result: dict[str, Any] = {
            "predicted_region": None,
            "vit_predicted_region": None,
            "source": "none",
            "confidence": 0.0,
            "streetclip_top5": [],
            "description_location": None,
        }

        # ── Layer 1: Vector DB (Image-to-Image) ────────────────────────────
        if HAS_TORCH:
            vector_result = self._layer_1_5_vector_db(image)
            if vector_result:
                result["predicted_region"] = vector_result["label"]
                result["source"] = "vector_db"
                result["confidence"] = vector_result["score"]
                logger.info(
                    "[geolocator] Layer 1 (Vector DB) hit: %s (%.4f)",
                    vector_result["label"],
                    vector_result["score"],
                )

        # ── Layer 1.5: ViT World Landmarks ─────────────────────────────────────
        if HAS_TORCH and not result["predicted_region"]:
            landmark_result = self._layer_1_vit_landmarks(image)
            if landmark_result:
                result["predicted_region"] = landmark_result["label"]
                result["vit_predicted_region"] = landmark_result["label"]
                result["source"] = "vit_landmarks"
                result["confidence"] = landmark_result["score"]
                logger.info(
                    "[geolocator] Layer 1.5 (ViT) hit: %s (%.4f)",
                    landmark_result["label"],
                    landmark_result["score"],
                )

        # ── Layer 2: StreetCLIP zero-shot ────────────────────────────────────
        if HAS_TORCH:
            streetclip_results = _classify_with_streetclip(image)
            result["streetclip_top5"] = streetclip_results

            if streetclip_results:
                top_hit = streetclip_results[0]
                logger.info(
                    "[geolocator] Layer 2 (StreetCLIP) top: %s (%.4f)",
                    top_hit["label"],
                    top_hit["score"],
                )
                # If StreetCLIP is confident enough, use it
                if top_hit["score"] > 0.35:
                    # If Layer 1 gave a result, compare and pick best
                    if result["predicted_region"] and result["confidence"] > top_hit["score"]:
                        # Trust StreetCLIP for Polish landmarks if ViT wasn't highly confident (>0.70)
                        if top_hit["label"] in POLISH_LANDMARKS:
                            result["predicted_region"] = top_hit["label"]
                            result["source"] = "streetclip"
                            result["confidence"] = top_hit["score"]
                            logger.info("[geolocator] Overriding ViT with StreetCLIP Polish landmark")
                    else:
                        result["predicted_region"] = top_hit["label"]
                        result["source"] = "streetclip"
                        result["confidence"] = top_hit["score"]

        return result

    def _layer_1_vit_landmarks(self, image: Image.Image) -> dict[str, Any] | None:
        """
        Layer 1: ViT world landmarks classifier.

        Returns the top prediction if confidence > threshold, else None.
        """
        land_proc, land_model = _load_vit_model(
            "mmgyorke/vit-world-landmarks", "landmarks"
        )

        if not land_proc or not land_model:
            return None

        device = _get_device()
        inputs = land_proc(images=image, return_tensors="pt")
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = land_model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze(0)
        score, idx = probs.max(dim=0)
        landmark_score = score.item()

        if landmark_score < 0.35:
            return None

        landmark_label = None
        if hasattr(land_model.config, "id2label"):
            landmark_label = land_model.config.id2label[idx.item()]

        if not landmark_label:
            return None

        return {"label": f"Landmark: {landmark_label}", "score": landmark_score}

    def _layer_1_5_vector_db(self, image: Image.Image) -> dict[str, Any] | None:
        """
        Layer 1.5: Compare image embeddings with local vector database.
        """
        if not _vector_db:
            return None
            
        bundle = _load_streetclip()
        if bundle is None:
            return None
            
        processor, model = bundle
        device = _get_device()
        
        inputs = processor(images=image, return_tensors="pt")
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            image_features = image_features.cpu().squeeze(0)
            
        best_label = None
        best_score = -1.0
        
        # Calculate cosine similarity
        for label, ref_vector in _vector_db.items():
            similarity = torch.nn.functional.cosine_similarity(image_features, ref_vector, dim=0).item()
            if similarity > best_score:
                best_score = similarity
                best_label = label
                
        # Cosine similarity threshold
        if best_label and best_score > 0.60:
            return {"label": best_label, "score": best_score}
            
        return None

    def calculate_distance(self, location_name: str, latitude: float, longitude: float) -> tuple[bool, float | None]:
        """Geocode the location name and calculate distance to provided GPS coordinates."""
        coords = _geocode_via_photon(location_name)
        if coords:
            pred_lat, pred_lon = coords
            distance_km = round(haversine_distance(latitude, longitude, pred_lat, pred_lon), 2)
            return (distance_km < 50.0), distance_km
        return True, None

    def predict_region(self, image_paths: list[str]) -> str | None:
        """
        Legacy interface — predict the geographic region depicted in images.

        Delegates to the 3-layer prediction system.
        """
        result = self._predict_region_multilayer(image_paths)
        return result.get("predicted_region")
