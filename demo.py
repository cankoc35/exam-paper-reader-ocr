import datetime
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional

import cv2

os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import main as pipeline


def _copy_segment_steps(image_stem: str, output_dir: Path) -> None:
    segment_dir = ROOT_DIR / "segment-data"
    if not segment_dir.exists():
        return

    for image_path in segment_dir.iterdir():
        if image_path.is_file() and image_path.name.startswith(f"{image_stem}-"):
            shutil.copy2(image_path, output_dir / image_path.name)


def run_demo(image_path: Optional[str] = None) -> None:
    effective_path = Path(image_path) if image_path else pipeline.IMAGE_PATH
    image_bgr = cv2.imread(str(effective_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {effective_path}")

    cropped_page = pipeline.segment_page(
        image_bgr,
        image_name=effective_path.name,
        save_steps=True,
    )
    if cropped_page is None:
        print("No page contour found, nothing to OCR.")
        return

    result_obj = pipeline.run_ocr(cropped_page, image_name=effective_path.name)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{effective_path.stem}_{ts}"
    pipeline.save_predictions(result_obj, run_name=run_name)
    cleaned_predictions = pipeline.save_cleaned_predictions(result_obj, run_name=run_name)
    metrics = pipeline.evaluate_predictions(cleaned_predictions, pipeline.ANSWERS_PATH)
    pipeline.save_evaluation_metrics(metrics, run_name=run_name)

    demo_dir = ROOT_DIR / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    _copy_segment_steps(effective_path.stem, demo_dir)

    metrics_path = pipeline.PRED_DIR / f"{run_name}_metrics.json"
    if metrics_path.exists():
        shutil.copy2(metrics_path, demo_dir / metrics_path.name)

    print(f"Demo artifacts saved to {demo_dir}")


if __name__ == "__main__":
    run_demo(sys.argv[1] if len(sys.argv) > 1 else None)
