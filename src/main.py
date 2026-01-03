import datetime
import json
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from segmentation import segment_page
from models import predict_trocr_both, run_easyocr, get_paddle_items

# predictions/ next to src/
PRED_DIR = Path(__file__).resolve().parent.parent / "predictions"
IMAGE_PATH = Path(__file__).resolve().parent.parent / "data" / "exam90.png"
ANSWERS_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "ground-truth-labels" / "answers.json"
)

def _order_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype="float32")
    point_sum = points.sum(axis=1)
    ordered[0] = points[np.argmin(point_sum)]  # top-left
    ordered[2] = points[np.argmax(point_sum)]  # bottom-right
    point_diff = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(point_diff)]  # top-right
    ordered[3] = points[np.argmax(point_diff)]  # bottom-left
    return ordered

def _polygon_bounds(polygon: np.ndarray) -> tuple[int, int, int, int]:
    x_values = polygon[:, 0]
    y_values = polygon[:, 1]
    return int(x_values.min()), int(y_values.min()), int(x_values.max()), int(y_values.max())

def _clip_box_to_image(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x_min = max(0, min(x_min, image_width - 1))
    x_max = max(0, min(x_max, image_width))
    y_min = max(0, min(y_min, image_height - 1))
    y_max = max(0, min(y_max, image_height))
    return x_min, y_min, x_max, y_max

def crop_text_region(image: np.ndarray, polygon: np.ndarray) -> Optional[np.ndarray]:
    if polygon.shape != (4, 2):
        return None

    image_height, image_width = image.shape[:2]
    polygon = polygon.astype("float32")
    polygon[:, 0] = np.clip(polygon[:, 0], 0, image_width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, image_height - 1)
    rect = _order_points(polygon)

    (tl, tr, br, bl) = rect
    width_bottom = np.linalg.norm(br - bl)
    width_top = np.linalg.norm(tr - tl)
    max_width = int(max(width_bottom, width_top))
    height_right = np.linalg.norm(tr - br)
    height_left = np.linalg.norm(tl - bl)
    max_height = int(max(height_right, height_left))

    if max_width < 2 or max_height < 2:
        return None

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(rect, destination)
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
    text_items = get_paddle_items(image)
    results = []

    # sort lines roughly top-to-bottom
    text_items = sorted(text_items, key=lambda it: np.array(it["poly"])[:, 1].mean())

    image_height, image_width = image.shape[:2]

    for line_index, item in enumerate(text_items):
        polygon = np.array(item["poly"])
        x_min, y_min, x_max, y_max = _polygon_bounds(polygon)
        x_min, y_min, x_max, y_max = _clip_box_to_image(
            x_min, y_min, x_max, y_max, image_width, image_height
        )

        if x_max <= x_min or y_max <= y_min:
            continue

        cropped_region = crop_text_region(image, polygon)
        if cropped_region is None:
            cropped_region = image[y_min:y_max, x_min:x_max]

        trocr_text = predict_trocr_both(cropped_region)
        easyocr_text = run_easyocr(cropped_region)

        results.append({
            "line_id": line_index,
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
    
def _extract_question_number(text: str) -> Optional[str]:
    match = re.search(r"\b(\d{1,2})\s*[).]", text)
    if not match:
        return None
    value = int(match.group(1))
    if 1 <= value <= 20:
        return str(value)
    return None

def _get_text_candidates(line: dict) -> list[str]:
    candidates = [line.get("paddle_text", "")]
    trocr_text = line.get("trocr_text", {})
    if isinstance(trocr_text, dict):
        candidates.extend([trocr_text.get("base", ""), trocr_text.get("large", "")])
    elif isinstance(trocr_text, str):
        candidates.append(trocr_text)
    easyocr = line.get("easyocr", [])
    if isinstance(easyocr, list):
        candidates.extend(easyocr)
    return candidates

def _extract_answer_variants(text: str) -> list[str]:
    if not text:
        return []
    match = re.match(r"^\s*\d{1,2}\s*[).]\s*(.+)$", text)
    base = match.group(1).strip() if match else text.strip()
    variants = [base] if base else []
    for separator in ("=", ":"):
        if separator in base:
            rhs = base.split(separator)[-1].strip()
            if rhs and rhs not in variants:
                variants.append(rhs)
    return variants

def _normalize_answer_text(text: str) -> str:
    return text.lower()

def _contains_match(expected: str, candidate: str) -> bool:
    if not expected or not candidate:
        return False
    return expected in candidate

def _levenshtein_distance(source: str, target: str) -> int:
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    if len(source) > len(target):
        source, target = target, source

    previous = list(range(len(source) + 1))
    for target_index, target_char in enumerate(target, start=1):
        current = [target_index]
        for source_index, source_char in enumerate(source, start=1):
            cost = 0 if source_char == target_char else 1
            current.append(min(
                previous[source_index] + 1,
                current[source_index - 1] + 1,
                previous[source_index - 1] + cost,
            ))
        previous = current
    return previous[-1]

def _candidate_answers_for_model(entry: dict, model: str) -> list[str]:
    candidates: list[str] = []
    followup = entry.get("followup") if isinstance(entry.get("followup"), dict) else None
    followups = entry.get("followups") if isinstance(entry.get("followups"), list) else []

    def add_text(value: str) -> None:
        for answer in _extract_answer_variants(value):
            candidates.append(answer)

    if model == "paddle":
        add_text(entry.get("paddle_text", ""))
        if followup:
            add_text(followup.get("paddle_text", ""))
        for extra in followups:
            if isinstance(extra, dict):
                add_text(extra.get("paddle_text", ""))
        return candidates

    if model == "trocr_base":
        trocr_text = entry.get("trocr_text", {})
        if isinstance(trocr_text, dict):
            add_text(trocr_text.get("base", ""))
        elif isinstance(trocr_text, str):
            add_text(trocr_text)
        if followup:
            followup_text = followup.get("trocr_text", {})
            if isinstance(followup_text, dict):
                add_text(followup_text.get("base", ""))
            elif isinstance(followup_text, str):
                add_text(followup_text)
        for extra in followups:
            if not isinstance(extra, dict):
                continue
            extra_text = extra.get("trocr_text", {})
            if isinstance(extra_text, dict):
                add_text(extra_text.get("base", ""))
            elif isinstance(extra_text, str):
                add_text(extra_text)
        return candidates

    if model == "trocr_large":
        trocr_text = entry.get("trocr_text", {})
        if isinstance(trocr_text, dict):
            add_text(trocr_text.get("large", ""))
        elif isinstance(trocr_text, str):
            add_text(trocr_text)
        if followup:
            followup_text = followup.get("trocr_text", {})
            if isinstance(followup_text, dict):
                add_text(followup_text.get("large", ""))
            elif isinstance(followup_text, str):
                add_text(followup_text)
        for extra in followups:
            if not isinstance(extra, dict):
                continue
            extra_text = extra.get("trocr_text", {})
            if isinstance(extra_text, dict):
                add_text(extra_text.get("large", ""))
            elif isinstance(extra_text, str):
                add_text(extra_text)
        return candidates

    if model == "easyocr":
        easyocr = entry.get("easyocr", [])
        if isinstance(easyocr, list):
            for item in easyocr:
                add_text(item)
        else:
            add_text(str(easyocr))
        if followup:
            followup_easyocr = followup.get("easyocr", [])
            if isinstance(followup_easyocr, list):
                for item in followup_easyocr:
                    add_text(item)
            else:
                add_text(str(followup_easyocr))
        for extra in followups:
            if not isinstance(extra, dict):
                continue
            extra_easyocr = extra.get("easyocr", [])
            if isinstance(extra_easyocr, list):
                for item in extra_easyocr:
                    add_text(item)
            else:
                add_text(str(extra_easyocr))
        return candidates

    return candidates

def save_cleaned_predictions(result_obj: dict, run_name: str) -> dict:
    cleaned_lines = []
    lines = sorted(result_obj.get("lines", []), key=lambda item: item.get("line_id", 0))
    index = 0
    while index < len(lines):
        line = lines[index]
        candidates = _get_text_candidates(line)

        question_number = None
        for text in candidates:
            question_number = _extract_question_number(text)
            if question_number:
                break

        if question_number:
            cleaned_entry = {
                "question": question_number,
                "paddle_text": line.get("paddle_text"),
                "trocr_text": line.get("trocr_text"),
                "easyocr": line.get("easyocr"),
            }

            followups = []
            next_index = index + 1
            while next_index < len(lines):
                next_line = lines[next_index]
                next_candidates = _get_text_candidates(next_line)
                next_question = None
                for text in next_candidates:
                    next_question = _extract_question_number(text)
                    if next_question:
                        break
                if next_question is not None:
                    break
                followups.append({
                    "paddle_text": next_line.get("paddle_text"),
                    "trocr_text": next_line.get("trocr_text"),
                    "easyocr": next_line.get("easyocr"),
                })
                next_index += 1

            if followups:
                cleaned_entry["followups"] = followups

            cleaned_lines.append(cleaned_entry)
            index = next_index
            continue

        index += 1

    cleaned_predictions = {
        "run_name": run_name,
        "image_name": result_obj.get("image_name"),
        "results": cleaned_lines,
    }

    out_path = PRED_DIR / f"{run_name}_cleaned.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_predictions, f, ensure_ascii=False, indent=2)

    print(f"Saved cleaned predictions to {out_path}")
    return cleaned_predictions

def evaluate_predictions(cleaned_predictions: dict, answers_path: Path) -> dict:
    with answers_path.open("r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    models = ["paddle", "trocr_base", "trocr_large", "easyocr"]
    predicted_by_model = {model: {} for model in models}
    for entry in cleaned_predictions.get("results", []):
        question = entry.get("question")
        if not question:
            continue
        for model in models:
            predicted_by_model[model][question] = _candidate_answers_for_model(entry, model)

    total_questions = len(ground_truth)
    model_metrics = {}

    for model in models:
        total_edits = 0
        total_chars = 0
        correct = 0
        predicted = predicted_by_model[model]

        for question, expected in ground_truth.items():
            predicted_candidates = predicted.get(question, [])
            expected_norm = _normalize_answer_text(expected)
            candidate_norms = [_normalize_answer_text(text) for text in predicted_candidates if text]

            if expected_norm and any(
                _contains_match(expected_norm, candidate) for candidate in candidate_norms
            ):
                correct += 1

            if candidate_norms:
                best_edits = min(
                    _levenshtein_distance(candidate, expected_norm) for candidate in candidate_norms
                )
            else:
                best_edits = len(expected_norm)

            total_edits += best_edits
            total_chars += len(expected_norm)

        cer = (total_edits / total_chars) if total_chars else 0.0
        accuracy = (correct / total_questions) if total_questions else 0.0
        model_metrics[model] = {
            "cer": cer,
            "accuracy": accuracy,
            "correct": correct,
            "total": total_questions,
        }

    return {
        "run_name": cleaned_predictions.get("run_name"),
        "image_name": cleaned_predictions.get("image_name"),
        "models": model_metrics,
    }

def save_evaluation_metrics(metrics: dict, run_name: str) -> None:
    out_path = PRED_DIR / f"{run_name}_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation metrics to {out_path}")


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

    image_bgr = cv2.imread(str(effective_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {effective_path}")

    cropped_page = segment_page(image_bgr, image_name=effective_path.name)
    if cropped_page is None:
        print("No page contour found, nothing to OCR.")
        return

    result_obj = run_ocr(cropped_page, image_name=effective_path.name)

    # use image stem plus timestamp so each run is unique
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{effective_path.stem}_{ts}"
    save_predictions(result_obj, run_name=run_name)
    cleaned_predictions = save_cleaned_predictions(result_obj, run_name=run_name)
    metrics = evaluate_predictions(cleaned_predictions, ANSWERS_PATH)
    save_evaluation_metrics(metrics, run_name=run_name)


if __name__ == "__main__":
    main()
