from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_SCENES = ROOT / "data" / "scenes"
REPORT_DIR = ROOT / "report"
FIGURES_DIR = REPORT_DIR / "figures"
RESULTS_DIR = REPORT_DIR / "results"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_scenes() -> list[Path]:
    if not DATA_SCENES.exists():
        return []
    scenes = [p for p in sorted(DATA_SCENES.iterdir()) if p.is_dir() and list_images(p)]
    return scenes


def list_images(folder: Path) -> list[Path]:
    files = [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
    ]
    return files
