# segmentation.py
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

def segment_page(
    img_bgr: np.ndarray,
    return_debug: bool = False,
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    # Segment the main page area from the input image.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE and Gaussian blur
    # to improve contrast and reduce noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    if return_debug:
        dbg = img_bgr.copy()
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 3)
        return cropped, dbg

    return cropped

if __name__ == "__main__":
    TEST_IMAGE_PATH = Path(__file__).resolve().parent.parent / "data" / "exam06.jpeg"
    img = cv2.imread(str(TEST_IMAGE_PATH))
    if img is None:
        raise FileNotFoundError(TEST_IMAGE_PATH)

    cropped, dbg = segment_page(img, return_debug=True)

    if cropped is not None:
        cv2.namedWindow("Segmented Page", cv2.WINDOW_NORMAL)
        cv2.imshow("Segmented Page", cropped)  # <- show the cropped image
        cv2.namedWindow("Page Overlay", cv2.WINDOW_NORMAL)
        cv2.imshow("Page Overlay", dbg)  # <- show the original with green box
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Segmentation failed: no page area found.")
