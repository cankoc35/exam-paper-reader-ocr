# SEDS536 Term Project: Exam Paper Reader & OCR

> **Course**: SEDS536 - Image Understanding (Master's Level)  
> **Author**: Mustafacan Koç  
> **Academic Year**: 2024-2025

## Project Overview

This project builds an exam paper reader that crops scanned pages, detects handwritten or printed text lines, and runs multiple OCR engines to extract answers in Turkish/Latin script. The goal is to experiment with detection/recognition combinations to improve accuracy on real student exam sheets.

### Objectives

- Robustly crop and deskew exam pages before OCR
- Detect text regions/lines and handle handwriting or mixed scripts
- Compare multiple recognizers (PaddleOCR, TrOCR, EasyOCR; optional Tesseract/DocTR)
- Add light language-aware cleanup for Turkish/Latin output
- Document a repeatable pipeline for future fine-tuning

### Key Features

- Page segmentation: contour-based crop of the main exam page
- Detector + recognizer stack: PaddleOCR detection feeding TrOCR/EasyOCR recognition
- Turkish/Latin language focus with multilingual model options
- Modular OCR runners in `src/` for quick swaps and comparisons
- Roadmap-driven improvements (cleanup, better models, post-correction)

## Project Status

**Current Phase**: Prototyping & model comparison  
See `ROADMAP.md` for detailed next steps (better language models, preprocessing, post-correction).

## Repository Structure

```
exam-paper-reader-ocr/
├─ README.md                  # This file
├─ ROADMAP.md                 # Planned improvements and model trials
├─ requirements.txt           # Python dependencies
├─ data/                      # Sample exam images (e.g., example_exam_02.jpeg)
└─ src/                       # OCR pipeline code
   ├─ main.py                 # Runs detection + multiple recognizers
   ├─ segmentation.py         # Page crop/segmentation with OpenCV
   ├─ paddleOCR.py            # PaddleOCR detection and text output
   ├─ trOCR.py                # Microsoft TrOCR inference helper
   └─ easyOCR.py              # EasyOCR inference helper
```

## Technical Approach

- **Pipeline**: segment page → detect text regions (PaddleOCR) → crop lines → run multiple recognizers → aggregate outputs for comparison.
- **Preprocessing**: grayscale, blur, threshold, and morphological cleanup to tighten the crop; roadmap includes deskewing and gentle contrast normalization.
- **Recognizers**: PaddleOCR for detection + baseline text; TrOCR for handwriting-friendly recognition; EasyOCR as an extra reference. Tesseract/DocTR/MMOCR are noted as optional experiments (see `ROADMAP.md`).
- **Language Handling**: prioritize Turkish/Latin models; plan a light post-correction step with Turkish spell-check or LM.
- **Evaluation**: manual inspection of line-by-line outputs; future work includes small labeled sets for quantitative scoring and potential fine-tuning.

## Data & Samples

- Default demo image: `data/example_exam_02.jpeg` (used by `segmentation.py`).  
- Add your own scans to `data/` and update the path in `segmentation.py` or pass images directly into `main(image)`.

## Development Setup

1) Create venv: `python -m venv venv`  
2) Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux)  
3) Install deps: `pip install -r requirements.txt`

### Running the sample

From the repo root (after activation):  
`python src/main.py`  
This uses the default sample image and prints PaddleOCR, TrOCR, and EasyOCR outputs per detected line.

## Testing

- Manual verification of OCR outputs on sample exams.  
- TODO: add a small labeled set + scoring script for WER/CER on Turkish/Latin lines.

## License

Academic project for SEDS536 course.

## Acknowledgments

Course: SEDS536 - Image Understanding
