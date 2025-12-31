from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

def _prepare_output_dir(image_name):
    output_dir = Path(__file__).resolve().parent.parent / "segment-data"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(image_name).stem if image_name else "image"
    return output_dir, prefix

def _save_step_image(output_dir, prefix, step_name, image):
    cv2.imwrite(str(output_dir / f"{prefix}-{step_name}.png"), image)

# Step 1: convert to grayscale and estimate illumination with a large blur.
# Step 2: normalize by illumination to reduce shadows, then apply CLAHE for contrast.
# Step 3: blur lightly and return normalized images plus a size hint.
def _normalize_image(image_bgr):
    image_height, image_width = image_bgr.shape[:2]
    min_dim = min(image_height, image_width)

    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    illum_kernel = max(31, (min_dim // 20) | 1)
    illumination = cv2.GaussianBlur(grayscale, (illum_kernel, illum_kernel), 0)
    shadow_corrected = cv2.divide(grayscale, illumination, scale=255)
    shadow_corrected = cv2.normalize(shadow_corrected, None, 0, 255, cv2.NORM_MINMAX)
    shadow_corrected = shadow_corrected.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(shadow_corrected)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return shadow_corrected, enhanced, blurred, min_dim

# Step 1: compute a global threshold using Otsu on the normalized image.
# Step 2: convert the result to a binary mask.
# Step 3: return the mask for later scoring and cleanup.
def _make_otsu_mask(image):
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

# Step 1: choose a block size based on the image scale.
# Step 2: apply adaptive threshold to handle local lighting changes.
# Step 3: return the binary mask for scoring against the Otsu mask.
def _make_adaptive_mask(image, min_dim):
    adaptive_block = max(31, (min_dim // 20) | 1)
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_block,
        5,
    )

# Step 1: remove small speckles with an opening.
# Step 2: close gaps so the page border is more connected.
# Step 3: if the result is mostly white, fall back to a less aggressive mask.
def _cleanup_mask(mask: np.ndarray) -> np.ndarray:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    white_ratio = np.count_nonzero(closed) / closed.size
    contour_mask = closed
    if white_ratio > 0.9:
        opened_white_ratio = np.count_nonzero(opened) / opened.size
        contour_mask = opened if opened_white_ratio < white_ratio else mask

    return contour_mask

# Step 1: reject contours outside the allowed area range or with invalid perimeter.
# Step 2: approximate the contour and compute rectangularity with a rotated box.
# Step 3: penalize border-touching contours and return the score with the polygon.
def _score_contour(
    contour: np.ndarray,
    image_area: float,
    min_page_area: float,
    max_contour_area: float,
    border_margin: int,
    image_width: int,
    image_height: int,
) -> Optional[Tuple[float, np.ndarray]]:
    contour_area = cv2.contourArea(contour)
    if contour_area < min_page_area or contour_area > max_contour_area:
        return None

    contour_perimeter = cv2.arcLength(contour, True)
    if contour_perimeter <= 0:
        return None

    contour_polygon = cv2.approxPolyDP(contour, 0.02 * contour_perimeter, True)
    if len(contour_polygon) < 4:
        return None

    rect = cv2.minAreaRect(contour_polygon)
    rect_w, rect_h = rect[1]
    if rect_w < 1 or rect_h < 1:
        return None

    rect_area = rect_w * rect_h
    rectangularity = contour_area / rect_area
    if rectangularity < 0.5:
        return None

    rect_x, rect_y, rect_wb, rect_hb = cv2.boundingRect(contour_polygon)
    touches_border = (
        rect_x <= border_margin
        or rect_y <= border_margin
        or rect_x + rect_wb >= image_width - border_margin
        or rect_y + rect_hb >= image_height - border_margin
    )
    area_ratio = contour_area / image_area
    if touches_border and area_ratio > 0.9:
        return None

    score = rectangularity + (0.35 * area_ratio)
    if touches_border:
        score -= 0.15

    return score, contour_polygon

# Step 1: score each contour using the page-likeness heuristic.
# Step 2: track the highest scoring contour.
# Step 3: return the best score and its polygon.
def _best_contour(
    contours: list[np.ndarray],
    image_area: float,
    min_page_area: float,
    max_contour_area: float,
    border_margin: int,
    image_width: int,
    image_height: int,
) -> Tuple[float, Optional[np.ndarray]]:
    best_score = -1.0
    best_contour = None
    for contour in contours:
        scored = _score_contour(
            contour,
            image_area,
            min_page_area,
            max_contour_area,
            border_margin,
            image_width,
            image_height,
        )
        if scored is None:
            continue
        score, contour_polygon = scored
        if score > best_score:
            best_score = score
            best_contour = contour_polygon
    return best_score, best_contour

# Step 1: find contours in the mask.
# Step 2: score the best contour using the same heuristic as final selection.
# Step 3: return the score so masks can be compared.
def _score_mask(
    mask: np.ndarray,
    image_area: float,
    min_page_area: float,
    max_contour_area: float,
    border_margin: int,
    image_width: int,
    image_height: int,
) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    score, _ = _best_contour(
        contours,
        image_area,
        min_page_area,
        max_contour_area,
        border_margin,
        image_width,
        image_height,
    )
    return score

# Step 1: normalize the image for stable processing.
# Step 2: run edge-based and mask-based pipelines in parallel and score their contours.
# Step 3: pick the higher-scoring contour and crop the page from the original image.
# Step 4: save debug output and return the cropped page (plus overlay if requested).
def segment_page(
    image_bgr: np.ndarray,
    return_debug: bool = False,
    image_name: Optional[str] = None,
    save_steps: bool = True,
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    output_dir = None
    prefix = None
    if save_steps:
        output_dir, prefix = _prepare_output_dir(image_name)

    image_height, image_width = image_bgr.shape[:2]

    shadow_corrected, enhanced_contrast, blurred_image, min_dim = _normalize_image(image_bgr)
    image_area = image_height * image_width
    min_page_area = 0.35 * image_area
    max_contour_area = 0.9 * image_area
    border_margin = max(5, int(0.01 * min_dim))

    otsu_binary = _make_otsu_mask(blurred_image)
    adaptive_binary = _make_adaptive_mask(enhanced_contrast, min_dim)
    otsu_score = _score_mask(
        otsu_binary,
        image_area,
        min_page_area,
        max_contour_area,
        border_margin,
        image_width,
        image_height,
    )
    adaptive_score = _score_mask(
        adaptive_binary,
        image_area,
        min_page_area,
        max_contour_area,
        border_margin,
        image_width,
        image_height,
    )
    base_binary = adaptive_binary if adaptive_score > otsu_score else otsu_binary
    contour_binary = _cleanup_mask(base_binary)

    # 4) Find the largest plausible page contour and crop it.
    page_contours, _ = cv2.findContours(
        contour_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if save_steps and output_dir and prefix:
        _save_step_image(output_dir, prefix, "step1-shadow-corrected", shadow_corrected)
        _save_step_image(output_dir, prefix, "step2-mask", base_binary)
        _save_step_image(output_dir, prefix, "step3-contour", contour_binary)

    median_val = float(np.median(blurred_image))
    lower = int(max(0, 0.66 * median_val))
    upper = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(blurred_image, lower, upper)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, edge_kernel, iterations=1)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_kernel, iterations=1)
    edge_contours, _ = cv2.findContours(
        edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    edge_score, edge_contour = _best_contour(
        edge_contours,
        image_area,
        min_page_area,
        max_contour_area,
        border_margin,
        image_width,
        image_height,
    )

    mask_score, mask_contour = _best_contour(
        page_contours,
        image_area,
        min_page_area,
        max_contour_area,
        border_margin,
        image_width,
        image_height,
    )

    best_page_contour = edge_contour
    if mask_contour is not None and (edge_contour is None or mask_score > edge_score):
        best_page_contour = mask_contour

    if best_page_contour is None and edge_contours:
        rect = cv2.minAreaRect(max(edge_contours, key=cv2.contourArea))
        best_page_contour = cv2.boxPoints(rect).astype(np.int32)

    if best_page_contour is None:
        if return_debug:
            return None, None
        return None

    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(best_page_contour)
    cropped_page = image_bgr[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]

    if return_debug or save_steps:
        debug_overlay = image_bgr.copy()
        cv2.rectangle(
            debug_overlay,
            (rect_x, rect_y),
            (rect_x + rect_w, rect_y + rect_h),
            (0, 255, 0),
            3,
        )
        if save_steps and output_dir and prefix:
            _save_step_image(output_dir, prefix, "step4-overlay", debug_overlay)
            _save_step_image(output_dir, prefix, "step4-cropped", cropped_page)
        if return_debug:
            return cropped_page, debug_overlay

    return cropped_page

if __name__ == "__main__":
    import sys

    default_path = Path(__file__).resolve().parent.parent / "data" / "exam103.png"
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(image_path)

    cropped_page, debug_overlay = segment_page(
        image_bgr, return_debug=True, image_name=image_path.name
    )

    if cropped_page is not None:
        cv2.namedWindow("Segmented Page", cv2.WINDOW_NORMAL)
        cv2.imshow("Segmented Page", cropped_page)  # <- show the cropped image
        cv2.namedWindow("Page Overlay", cv2.WINDOW_NORMAL)
        cv2.imshow("Page Overlay", debug_overlay)  # <- show the original with green box
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Segmentation failed: no page area found.")
