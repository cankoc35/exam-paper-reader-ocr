import json
import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from segmentation import segment_page
from paddleOCR import get_paddle_items
from trOCR import predict_trocr_both
from easyOCR import run_easyocr

# predictions/ next to src/
PRED_DIR = Path(__file__).resolve().parent.parent / "predictions"
IMAGE_PATH = Path(__file__).resolve().parent.parent / "data" / "exam02.png"

def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def crop_text_region(image: np.ndarray, poly: np.ndarray) -> Optional[np.ndarray]:
    if poly.shape != (4, 2):
        return None

    h, w = image.shape[:2]
    poly = poly.astype("float32")
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    rect = _order_points(poly)

    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width < 2 or max_height < 2:
        return None

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, transform, (max_width, max_height))


def run_ocr(image: np.ndarray, image_name: Optional[str] = None):
    """
    Run OCR on a cropped page image and return a dict with metadata.

    image: cropped BGR page image (np.ndarray)
    image_name: original image file name (optional)
    """
    if image is None:
        print("No cropped image to process for OCR.")
        return {"image_name": image_name, "lines": []}

    # Paddle performs detection + recognition on text regions.
    items = get_paddle_items(image)
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

        crop = crop_text_region(image, poly)
        if crop is None:
            crop = image[y_min:y_max, x_min:x_max]

        trocr_text = predict_trocr_both(crop)
        easyocr_text = run_easyocr(crop)

        results.append({
            "line_id": i,
            "bbox": [x_min, y_min, x_max, y_max],
            "paddle_text": item["text"],
            "trocr_text": trocr_text,
            "easyocr": easyocr_text,
        }) 

    return {
        "image_name": image_name,
        "lines": results,
    }


def save_predictions(result_obj: dict, run_name: Optional[str] = None):
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


def main(img_path: Optional[str] = None):
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

    cropped = segment_page(img, image_name=effective_path.name)
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
