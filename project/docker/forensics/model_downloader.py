"""
model_downloader.py
Uruchamiany TYLKO podczas `docker build` — pobiera wagi modeli do /models.
Po build: HF_DATASETS_OFFLINE=1 blokuje wszelkie zewnętrzne wywołania.
"""
import os
import sys

CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", "/models")

MODELS = [
    # PRIMARY — SigLIP2-base, general-purpose AI vs human
    # Trenowany na 60k AI + 60k human images (Midjourney v6+, SD 3.5, GPT-4o)
    # ~330 MB, 99.2% test accuracy, obsługa CPU i GPU
    {
        "repo_id": "Ateeqq/ai-vs-human-image-detector",
        "processor_cls": "AutoImageProcessor",
        "model_cls": "SiglipForImageClassification",
    },
    # FAST FILTER — ViT distilled, 11.8M params
    # Distylacja 3 modeli (Midjourney, SD, SD fine-tunings)
    # ~45 MB, używany jako szybki pre-filter przed primary
    {
        "repo_id": "jacoballessio/ai-image-detect-distilled",
        "processor_cls": "AutoFeatureExtractor",
        "model_cls": "ViTForImageClassification",
    },
]


def download_models() -> None:
    from transformers.models.auto.feature_extraction_auto import AutoFeatureExtractor
    from transformers.models.auto.image_processing_auto import AutoImageProcessor
    from transformers.models.siglip import SiglipForImageClassification
    from transformers.models.vit import ViTForImageClassification

    cls_map = {
        "AutoImageProcessor": AutoImageProcessor,
        "AutoFeatureExtractor": AutoFeatureExtractor,
        "SiglipForImageClassification": SiglipForImageClassification,
        "ViTForImageClassification": ViTForImageClassification,
    }

    for entry in MODELS:
        repo_id = entry["repo_id"]
        print(f"[downloader] Pobieranie: {repo_id}", flush=True)
        try:
            processor_cls = cls_map[entry["processor_cls"]]
            model_cls = cls_map[entry["model_cls"]]

            processor_cls.from_pretrained(repo_id, cache_dir=CACHE_DIR)
            model_cls.from_pretrained(repo_id, cache_dir=CACHE_DIR)

            print(f"[downloader] OK: {repo_id}", flush=True)
        except Exception as exc:
            print(f"[downloader] BŁĄD ({repo_id}): {exc}", file=sys.stderr, flush=True)
            sys.exit(1)

    print("[downloader] Wszystkie modele pobrane.", flush=True)


if __name__ == "__main__":
    download_models()