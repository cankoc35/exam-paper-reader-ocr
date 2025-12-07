import json
import datetime
from pathlib import Path
import numpy as np

from segmentation import cropped_image
from paddleOCR import get_paddle_items
from trOCR import predict_trocr

PRED_DIR = Path(__file__).resolve().parent.parent / "predictions"


def run_ocr(image):
    """
    Run OCR on a cropped page image and return a list of dicts
    (one per detected line).
    """
    if image is None:
        print("No cropped image to process for OCR.")
        return []

    items = get_paddle_items(image)   # Paddle detection + text
    results = []

    for i, item in enumerate(items):
        poly = np.array(item["poly"])
        xs = poly[:, 0]
        ys = poly[:, 1]

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        crop = image[y_min:y_max, x_min:x_max]

        trocr_text = predict_trocr(crop)

        results.append({
            "line_id": i,
            "bbox": [x_min, y_min, x_max, y_max],
            "paddle_text": item["text"],
            "paddle_score": item["score"],
            "trocr_text": trocr_text,
        })

    return results


def save_predictions(results, run_name=None):
    """
    Save OCR results to predictions/<run_name>.json
    """
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"run_{ts}"

    out_path = PRED_DIR / f"{run_name}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"run_name": run_name, "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved predictions to {out_path}")


def main():
    results = run_ocr(cropped_image)
    save_predictions(results)


if __name__ == "__main__":
    main()
