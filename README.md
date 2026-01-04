# SEDS536 Exam Paper Reader & OCR

This project segments a raw exam photo to isolate the paper, runs OCR models on the text regions, and evaluates predictions against labeled answers.

## Pipeline (Short)
1) Segment the page (shadow correction + threshold/edge masks + contour crop).
2) Detect text regions with PaddleOCR.
3) Crop each region and run PaddleOCR + EasyOCR + TrOCR (base/large) for recognition.
4) Save raw + cleaned predictions and compute CER/accuracy per model.

## Methodology
1) **Page segmentation**: Normalize illumination, build mask/edge cues, and extract the page contour to crop a clean paper view.
2) **Text region detection**: Use PaddleOCR detection to locate line-level text polygons on the cropped page.
3) **Region normalization**: Warp each polygon to a rectangular crop so OCR models see upright text.
4) **Multi-model OCR**: Run PaddleOCR, EasyOCR, and TrOCR (base/large) on each crop for complementary predictions.
5) **Answer structuring**: Detect question numbers (1–20) per line, anchor bbox method per question.
6) **Evaluation**: Compare model answers with ground-truth labels to compute per-model accuracy and CER.

## Project Structure
```
exam-paper-reader-ocr/
|- README.md
|- requirements.txt
|- demo.py             # Demo runner (writes artifacts to demo/)
|- demo/               # Demo outputs (step images + metrics)
|- data/                # Input images
|- predictions/         # Raw/cleaned predictions + metrics
|- segment-data/        # Saved segmentation steps
|- docs/
   |- ground-truth-labels/
      |- answers.json   # Labeled answers for evaluation
|- src/
   |- main.py           # End-to-end pipeline
   |- segmentation.py   # Page segmentation + step images
   |- models.py         # PaddleOCR, EasyOCR, TrOCR
```

## Run
1) Create venv: `python -m venv venv`
2) Activate: `source venv/bin/activate`
3) Install deps: `pip install -r requirements.txt`
4) Run: `python src/main.py`

Default image path is set in `src/main.py` (`IMAGE_PATH`). Outputs are written to `predictions/` and step images to `segment-data/`.

## Batch Run 
Run the pipeline for every image in `data/`:

```
python - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, "src")
import main

data_dir = Path("data")
for img in sorted(data_dir.glob("*.png")):
    print(f"Processing {img.name}")
    main.main(str(img))
PY
```

## Demo
Run the end-to-end demo and collect artifacts in `demo/`:

```
python demo.py
```

Optional: pass a specific image path:
```
python demo.py data/exam68.png
```

The demo folder will include segmentation step images and the run metrics JSON for quick presentation.
