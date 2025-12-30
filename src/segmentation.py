# segmentation.py
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

def segment_page(
    img_bgr: np.ndarray,
    return_debug: bool = False,
    image_name: Optional[str] = None,
    save_steps: bool = True,
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    # 1) Convert to grayscale, then enhance contrast and blur for stability.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 2) Binarize with Otsu to separate page from background.
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3) Close gaps so the page contour becomes a single region.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4) Find the largest plausible page contour and crop it.
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if save_steps:
        out_dir = Path(__file__).resolve().parent.parent / "segment-data"
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = Path(image_name).stem if image_name else "image"
        cv2.imwrite(str(out_dir / f"{prefix}-step1-gray.png"), gray)
        cv2.imwrite(str(out_dir / f"{prefix}-step1-enhanced.png"), enhanced)
        cv2.imwrite(str(out_dir / f"{prefix}-step1-blurred.png"), blurred)
        cv2.imwrite(str(out_dir / f"{prefix}-step2-otsu.png"), th)
        cv2.imwrite(str(out_dir / f"{prefix}-step3-closed.png"), closed)

    h, w = img_bgr.shape[:2]
    min_area = 0.25 * h * w
    best = None
    
    # Find the largest contour that is likely to be the page
    # by area and shape criteria 
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) < 4:
            continue

        best = approx
        break

    if best is None:
        if return_debug:
            return None, None
        return None

    x, y, w, h = cv2.boundingRect(best)
    cropped = img_bgr[y:y+h, x:x+w]

    if return_debug or save_steps:
        dbg = img_bgr.copy()
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if save_steps:
            out_dir = Path(__file__).resolve().parent.parent / "segment-data"
            out_dir.mkdir(parents=True, exist_ok=True)
            prefix = Path(image_name).stem if image_name else "image"
            cv2.imwrite(str(out_dir / f"{prefix}-step4-overlay.png"), dbg)
            cv2.imwrite(str(out_dir / f"{prefix}-step4-cropped.png"), cropped)
        if return_debug:
            return cropped, dbg

    return cropped

if __name__ == "__main__":
    import sys

    default_path = Path(__file__).resolve().parent.parent / "data" / "exam03.png"
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)

    cropped, dbg = segment_page(img, return_debug=True, image_name="exam03.png")

    if cropped is not None:
        cv2.namedWindow("Segmented Page", cv2.WINDOW_NORMAL)
        cv2.imshow("Segmented Page", cropped)  # <- show the cropped image
        cv2.namedWindow("Page Overlay", cv2.WINDOW_NORMAL)
        cv2.imshow("Page Overlay", dbg)  # <- show the original with green box
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Segmentation failed: no page area found.")
