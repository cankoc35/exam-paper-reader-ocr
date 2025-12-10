import json
import datetime
from pathlib import Path

import cv2
import numpy as np

from segmentation import segment_page
from paddleOCR import get_paddle_items
from trOCR import predict_trocr

# predictions/ next to src/
PRED_DIR = Path(__file__).resolve().parent.parent / "predictions"
IMAGE_PATH = Path(__file__).resolve().parent.parent / "data" / "example_exam_06.jpeg"


def run_ocr(image: np.ndarray, image_name: str | None = None):
    """
    Run OCR on a cropped page image and return a dict with metadata.

    image: cropped BGR page image (np.ndarray)
    image_name: original image file name (optional)
    """
    if image is None:
        print("No cropped image to process for OCR.")
        return {"image_name": image_name, "lines": []}

    items = get_paddle_items(image)   # Paddle detection + text
    results = []

    # sort lines roughly top-to-bottom
    items = sorted(items, key=lambda it: np.array(it["poly"])[:, 1].mean())

    h, w = image.shape[:2]

    for i, item in enumerate(items):
        poly = np.array(item["poly"])
        xs = poly[:, 0]
        ys = poly[:, 1]

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        # safety clipping
        x_min = max(0, min(x_min, w - 1))
        x_max = max(0, min(x_max, w))
        y_min = max(0, min(y_min, h - 1))
        y_max = max(0, min(y_max, h))

        if x_max <= x_min or y_max <= y_min:
            continue

        crop = image[y_min:y_max, x_min:x_max]

        trocr_text = predict_trocr(crop)

        results.append({
            "line_id": i,
            "bbox": [x_min, y_min, x_max, y_max],
            "paddle_text": item["text"],
            "paddle_score": float(item["score"]),
            "trocr_text": trocr_text,
        })

    return {
        "image_name": image_name,
        "lines": results,
    }


def save_predictions(result_obj: dict, run_name: str | None = None):
    """
    Save OCR results to predictions/<run_name>.json
    result_obj: dict returned by run_ocr
    """
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"run_{ts}"

    out_path = PRED_DIR / f"{run_name}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "image_name": result_obj.get("image_name"),
                "results": result_obj.get("lines", []),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved predictions to {out_path}")


def main(img_path: str | None = None):
    """
    End-to-end:
    - read original exam image
    - segment page
    - run OCR
    - save JSON with image name
    """

    # pick provided path or default example
    effective_path = Path(IMAGE_PATH if img_path is None else img_path)

    img = cv2.imread(str(effective_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {effective_path}")

    cropped = segment_page(img)
    if cropped is None:
        print("No page contour found, nothing to OCR.")
        return

    result_obj = run_ocr(cropped, image_name=effective_path.name)

    # use image stem plus timestamp so each run is unique
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{effective_path.stem}_{ts}"
    save_predictions(result_obj, run_name=run_name)


if __name__ == "__main__":
    main()
