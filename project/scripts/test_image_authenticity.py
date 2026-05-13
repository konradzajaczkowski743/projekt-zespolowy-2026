"""Test lokalny: wczytaj obraz i zwróć JSON z procentową oceną autentyczności."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Użycie: python scripts/test_image_authenticity.py /ścieżka/do/obrazu.jpg",
            file=sys.stderr,
        )
        return 1

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Plik nie istnieje: {image_path}", file=sys.stderr)
        return 1

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    try:
        from ml_engine.services.forensics_advanced import analyze_image_from_path
    except Exception as exc:
        print(
            "Nie udało się zaimportować zaawansowanej analizy. "
            "Upewnij się, że masz zainstalowany PyTorch i transformers.",
            file=sys.stderr,
        )
        raise

    result = analyze_image_from_path(str(image_path))

    authenticity_percent = round(100.0 * (1.0 - result.confidence), 2)
    output = {
        "image_path": str(image_path.resolve()),
        "is_authentic": not result.is_ai_generated,
        "authenticity_percent": authenticity_percent,
        "ai_probability": round(result.confidence * 100, 2),
        "details": result.to_dict(),
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
