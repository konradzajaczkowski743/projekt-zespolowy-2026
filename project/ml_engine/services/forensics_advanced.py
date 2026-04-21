"""
Advanced forensics analysis module with ML-based AI/deepfake detection.

Architektura:
  Warstwa 1 — ELA  (Error Level Analysis)     — artefakty pikselowe, ~50ms CPU
  Warstwa 2 — EXIF (metadata consistency)     — spójność metadanych,  ~5ms
  Warstwa 3 — ML Ensemble                     — dwa modele HF,        ~500ms CPU / ~80ms GPU
    3a. Fast filter : jacoballessio/ai-image-detect-distilled  (11.8M params)
    3b. Primary     : Ateeqq/ai-vs-human-image-detector        (SigLIP2-base)

Wynik końcowy: ForensicsResult z polem is_ai_generated + confidence + details.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

logger = logging.getLogger(__name__)

# Warunkowy import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("[forensics] PyTorch nie zainstalowany — ML ensemble będzie zablokowany")

# Cache directory dla modeli HuggingFace
CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", "/tmp/models")

# Wagi ensemble — im wyższy wynik tym bardziej "AI generated"
_ENSEMBLE_WEIGHTS = {
    "fast_filter": 0.30,   # jacoballessio — lekki, mniej precyzyjny
    "primary":     0.70,   # Ateeqq SigLIP2 — główny model
}

# Próg powyżej którego obraz uznajemy za AI-generated
_DECISION_THRESHOLD = 0.55

# Jakość JPEG do analizy ELA (im niższa, tym bardziej widoczne artefakty)
_ELA_QUALITY = 75

# Znane tagi EXIF wskazujące na generowanie AI
_AI_SOFTWARE_TAGS = {
    "stable diffusion", "midjourney", "dall-e", "dalle",
    "firefly", "imagen", "flux", "automatic1111", "comfyui",
    "novelai", "invokeai",
}


# ── Struktury danych ──────────────────────────────────────────────────────────

@dataclass
class ELAResult:
    mean_error: float
    max_error: float
    suspicious_regions_pct: float   # % pikseli powyżej progu błędu


@dataclass
class EXIFResult:
    has_exif: bool
    ai_software_detected: bool
    suspicious_fields: list[str] = field(default_factory=list)
    software_tag: Optional[str] = None
    make_tag: Optional[str] = None


@dataclass
class MLResult:
    fast_filter_score: float        # 0.0 = real, 1.0 = AI
    primary_score: float
    ensemble_score: float           # ważona kombinacja
    fast_filter_label: str
    primary_label: str


@dataclass
class ForensicsResult:
    """Główny wynik zwracany do Django."""
    is_ai_generated: bool
    confidence: float               # 0.0–1.0, im wyższy tym pewniejszy wyrok

    ela: Optional[ELAResult] = None
    exif: Optional[EXIFResult] = None
    ml: Optional[MLResult] = None

    error: Optional[str] = None     # wypełniane tylko przy wyjątku

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_ai_generated": self.is_ai_generated,
            "confidence": round(self.confidence, 4),
            "ela": {
                "mean_error": round(self.ela.mean_error, 4),
                "max_error": round(self.ela.max_error, 4),
                "suspicious_regions_pct": round(self.ela.suspicious_regions_pct, 4),
            } if self.ela else None,
            "exif": {
                "has_exif": self.exif.has_exif,
                "ai_software_detected": self.exif.ai_software_detected,
                "suspicious_fields": self.exif.suspicious_fields,
                "software_tag": self.exif.software_tag,
                "make_tag": self.exif.make_tag,
            } if self.exif else None,
            "ml": {
                "fast_filter_score": round(self.ml.fast_filter_score, 4),
                "primary_score": round(self.ml.primary_score, 4),
                "ensemble_score": round(self.ml.ensemble_score, 4),
                "fast_filter_label": self.ml.fast_filter_label,
                "primary_label": self.ml.primary_label,
            } if self.ml else None,
            "error": self.error,
        }


# ── Lazy-loading modeli (singleton per proces) ────────────────────────────────

_models_cache: dict = {}


def _get_device() -> Optional[Any]:
    """Auto-detect: CUDA > CPU. Zwraca None jeśli torch niedostępny."""
    if not HAS_TORCH:
        return None
    if torch.cuda.is_available():  # type: ignore[name-defined]
        logger.info("[forensics] Używam GPU (CUDA)")
        return torch.device("cuda")  # type: ignore[name-defined]
    logger.info("[forensics] Używam CPU")
    return torch.device("cpu")  # type: ignore[name-defined]


def _load_fast_filter():
    if "fast_filter" not in _models_cache:
        if not HAS_TORCH:
            logger.warning("[forensics] PyTorch niedostępny — pominięcie fast_filter")
            return None, None

        from transformers.models.auto.feature_extraction_auto import AutoFeatureExtractor
        from transformers.models.vit import ViTForImageClassification

        logger.info("[forensics] Ładowanie fast_filter (jacoballessio)...")
        repo = "jacoballessio/ai-image-detect-distilled"
        try:
            _models_cache["fast_filter_processor"] = AutoFeatureExtractor.from_pretrained(
                repo, cache_dir=CACHE_DIR
            )
            model = ViTForImageClassification.from_pretrained(
                repo, cache_dir=CACHE_DIR
            )
            device = _get_device()
            if device is not None:
                model = model.to(device)  # type: ignore[union-attr]
            _models_cache["fast_filter"] = model.eval()
            logger.info("[forensics] fast_filter załadowany.")
        except Exception as exc:
            logger.exception("[forensics] Błąd ładowania fast_filter: %s", exc)
            _models_cache["fast_filter"] = None
            _models_cache["fast_filter_processor"] = None

    return _models_cache.get("fast_filter_processor"), _models_cache.get("fast_filter")


def _load_primary():
    if "primary" not in _models_cache:
        if not HAS_TORCH:
            logger.warning("[forensics] PyTorch niedostępny — pominięcie primary")
            return None, None

        from transformers.models.auto.image_processing_auto import AutoImageProcessor
        from transformers.models.siglip import SiglipForImageClassification

        logger.info("[forensics] Ładowanie primary (Ateeqq SigLIP2)...")
        repo = "Ateeqq/ai-vs-human-image-detector"
        try:
            _models_cache["primary_processor"] = AutoImageProcessor.from_pretrained(
                repo, cache_dir=CACHE_DIR
            )
            model = SiglipForImageClassification.from_pretrained(
                repo, cache_dir=CACHE_DIR
            )
            device = _get_device()
            if device is not None:
                model = model.to(device)  # type: ignore[union-attr]
            _models_cache["primary"] = model.eval()
            logger.info("[forensics] primary załadowany.")
        except Exception as exc:
            logger.exception("[forensics] Błąd ładowania primary: %s", exc)
            _models_cache["primary"] = None
            _models_cache["primary_processor"] = None

    return _models_cache.get("primary_processor"), _models_cache.get("primary")


# ── Warstwa 1: ELA ────────────────────────────────────────────────────────────

def _run_ela(image: Image.Image) -> ELAResult:
    """
    Error Level Analysis — wykrywa artefakty resamplingu i compositing.
    """
    img_rgb = image.convert("RGB")

    buffer = io.BytesIO()
    img_rgb.save(buffer, format="JPEG", quality=_ELA_QUALITY)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")

    diff = ImageChops.difference(img_rgb, compressed)
    diff_enhanced = ImageEnhance.Brightness(diff).enhance(10)

    diff_arr = np.array(diff_enhanced, dtype=np.float32)
    mean_error = float(diff_arr.mean())
    max_error = float(diff_arr.max())

    threshold = 30.0
    suspicious_pct = float((diff_arr > threshold).mean()) * 100.0

    return ELAResult(
        mean_error=mean_error,
        max_error=max_error,
        suspicious_regions_pct=suspicious_pct,
    )


# ── Warstwa 2: EXIF ───────────────────────────────────────────────────────────

def _run_exif(image: Image.Image) -> EXIFResult:
    """
    Sprawdza metadane EXIF pod kątem podejrzanych znaków AI.
    """
    suspicious_fields: list[str] = []
    software_tag: Optional[str] = None
    make_tag: Optional[str] = None
    ai_software_detected = False

    try:
        exif_data = image._getexif()  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        exif_data = None

    has_exif = exif_data is not None and len(exif_data) > 0

    if not has_exif:
        suspicious_fields.append("brak_exif")
        return EXIFResult(
            has_exif=False,
            ai_software_detected=False,
            suspicious_fields=suspicious_fields,
        )

    # Tag IDs: 271=Make, 272=Model, 305=Software
    TAG_MAKE = 271
    TAG_MODEL = 272
    TAG_SOFTWARE = 305

    if not isinstance(exif_data, dict):
        suspicious_fields.append("brak_exif")
        return EXIFResult(
            has_exif=False,
            ai_software_detected=False,
            suspicious_fields=suspicious_fields,
        )

    make_tag = exif_data.get(TAG_MAKE)
    model_tag = exif_data.get(TAG_MODEL)
    software_tag = exif_data.get(TAG_SOFTWARE)

    if not make_tag:
        suspicious_fields.append("brak_make")
    if not model_tag:
        suspicious_fields.append("brak_model")

    if software_tag:
        sw_lower = software_tag.lower()
        for ai_sw in _AI_SOFTWARE_TAGS:
            if ai_sw in sw_lower:
                ai_software_detected = True
                suspicious_fields.append(f"ai_software:{software_tag}")
                break

    return EXIFResult(
        has_exif=True,
        ai_software_detected=ai_software_detected,
        suspicious_fields=suspicious_fields,
        software_tag=software_tag,
        make_tag=make_tag,
    )


# ── Warstwa 3: ML Ensemble ────────────────────────────────────────────────────

def _score_to_ai_probability(logits: Any, model_labels: dict) -> tuple[float, str]:
    """Normalizuje wyjście modelu do prawdopodobieństwa AI (0.0–1.0)."""
    if not HAS_TORCH:
        return 0.5, "UNKNOWN"

    probs = torch.softmax(logits, dim=-1).squeeze()  # type: ignore[name-defined]
    probs_list = probs.tolist()
    if isinstance(probs_list, float):
        probs_list = [probs_list]

    ai_keywords = {"ai", "fake", "artificial", "generated", "0"}
    real_keywords = {"real", "human", "hum", "1"}

    ai_idx = None
    real_idx = None

    for idx, label in model_labels.items():
        label_lower = str(label).lower()
        if any(kw in label_lower for kw in ai_keywords):
            ai_idx = idx
        if any(kw in label_lower for kw in real_keywords):
            real_idx = idx

    if ai_idx is None:
        ai_idx = 0

    ai_prob = float(probs_list[ai_idx]) if ai_idx < len(probs_list) else 0.5
    label = "AI" if ai_prob >= 0.5 else "REAL"

    return ai_prob, label


def _run_ml_ensemble(image: Image.Image) -> Optional[MLResult]:
    """Krok 1: Fast filter. Krok 2: Primary. Wynik: ważona suma."""
    if not HAS_TORCH:
        logger.warning("[forensics] PyTorch niedostępny — pominięcie ML ensemble")
        return None

    device = _get_device()
    img_rgb = image.convert("RGB")

    # ── Fast filter ───────────────────────────────────────────────────────────
    ff_processor, ff_model = _load_fast_filter()
    if not ff_model or not ff_processor:
        logger.warning("[forensics] fast_filter niedostępny")
        ff_score = 0.5
        ff_label = "UNKNOWN"
    else:
        ff_inputs = ff_processor(images=img_rgb, return_tensors="pt")  # type: ignore[operator]
        if device is not None:
            ff_inputs = {k: v.to(device) for k, v in ff_inputs.items()}

        with torch.no_grad():  # type: ignore[name-defined]
            ff_outputs = ff_model(**ff_inputs)

        ff_score, ff_label = _score_to_ai_probability(
            ff_outputs.logits, ff_model.config.id2label
        )

    # ── Primary ───────────────────────────────────────────────────────────────
    pr_processor, pr_model = _load_primary()
    if not pr_model or not pr_processor:
        logger.warning("[forensics] primary niedostępny")
        pr_score = 0.5
        pr_label = "UNKNOWN"
    else:
        pr_inputs = pr_processor(images=img_rgb, return_tensors="pt")  # type: ignore[operator]
        if device is not None:
            pr_inputs = {k: v.to(device) for k, v in pr_inputs.items()}

        with torch.no_grad():  # type: ignore[name-defined]
            pr_outputs = pr_model(**pr_inputs)

        pr_score, pr_label = _score_to_ai_probability(
            pr_outputs.logits, pr_model.config.id2label
        )

    # ── Ensemble ──────────────────────────────────────────────────────────────
    ensemble_score = (
        _ENSEMBLE_WEIGHTS["fast_filter"] * ff_score
        + _ENSEMBLE_WEIGHTS["primary"] * pr_score
    )

    return MLResult(
        fast_filter_score=ff_score,
        primary_score=pr_score,
        ensemble_score=ensemble_score,
        fast_filter_label=ff_label,
        primary_label=pr_label,
    )


# ── Scoring końcowy ───────────────────────────────────────────────────────────

def _compute_final_score(
    ela: ELAResult,
    exif: EXIFResult,
    ml: Optional[MLResult],
) -> tuple[bool, float]:
    """Łączy sygnały z trzech warstw w jeden werdykt."""
    # ML score — już 0.0–1.0
    ml_signal = ml.ensemble_score if ml else 0.5

    # ELA score — normalizacja
    ela_signal = min(ela.mean_error / 25.0, 1.0)

    # EXIF score
    exif_signal = 0.0
    if not exif.has_exif:
        exif_signal = 0.4
    if exif.ai_software_detected:
        exif_signal = 1.0

    # Ważona suma
    final_score = (
        0.70 * ml_signal
        + 0.20 * ela_signal
        + 0.10 * exif_signal
    )

    # EXIF override
    if exif.ai_software_detected:
        final_score = max(final_score, 0.90)

    is_ai = final_score >= _DECISION_THRESHOLD
    return is_ai, float(final_score)


# ── Publiczne API ─────────────────────────────────────────────────────────────

def analyze_image(image: Image.Image) -> ForensicsResult:
    """Główna funkcja — wywołaj z Django."""
    try:
        ela_result = _run_ela(image)
        logger.debug("[forensics] ELA: mean=%.2f suspicious=%.1f%%",
                     ela_result.mean_error, ela_result.suspicious_regions_pct)

        exif_result = _run_exif(image)
        logger.debug("[forensics] EXIF: has=%s ai_sw=%s fields=%s",
                     exif_result.has_exif,
                     exif_result.ai_software_detected,
                     exif_result.suspicious_fields)

        ml_result = _run_ml_ensemble(image)
        if ml_result:
            logger.debug("[forensics] ML: ff=%.3f primary=%.3f ensemble=%.3f",
                         ml_result.fast_filter_score,
                         ml_result.primary_score,
                         ml_result.ensemble_score)

        is_ai, confidence = _compute_final_score(ela_result, exif_result, ml_result)

        return ForensicsResult(
            is_ai_generated=is_ai,
            confidence=confidence,
            ela=ela_result,
            exif=exif_result,
            ml=ml_result,
        )

    except Exception as exc:
        logger.exception("[forensics] Błąd analizy: %s", exc)
        return ForensicsResult(
            is_ai_generated=False,
            confidence=0.0,
            error=str(exc),
        )


def analyze_image_from_bytes(data: bytes) -> ForensicsResult:
    """Wygodny wrapper dla surowych bajtów."""
    image = Image.open(io.BytesIO(data))
    return analyze_image(image)


def analyze_image_from_path(path: str) -> ForensicsResult:
    """Wygodny wrapper dla ścieżki do pliku."""
    image = Image.open(path)
    return analyze_image(image)
