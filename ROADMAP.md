# Roadmap

Structured plan to improve OCR quality on Turkish/Latin exam pages. Follow the phases in order; add optional experiments as time permits.

## Phase 1: Input cleanup and detection
- Crop and deskew pages; tighten margins to remove shadows and borders.
- Light preprocessing before OCR: grayscale, gentle contrast normalization, mild denoise (avoid harsh binarization).
- Stick with multilingual/Latin-friendly models; avoid English-only configs.
- Re-run PaddleOCR detection after cleanup and verify bounding boxes visually.

## Phase 2: Line-centric recognition
- Use a handwriting-friendly recognizer (TrOCR, PARSeq) instead of default PP-OCRv4 mobile for cursive lines.
- Process line-by-line: detect text lines (PaddleOCR detector or projection), crop, slightly upscale (e.g., 2x via `cv2.resize`), then recognize.
- Keep PaddleOCR text output as a baseline; compare against TrOCR/EasyOCR on the same line crops.

## Phase 3: Language-aware post-processing
- Add Turkish post-correction (spellchecker or lightweight language model) to fix obvious errors.
- Keep a list of common OCR confusions (e.g., g/ğ, s/ş, i/ı) and normalize them.
- Evaluate outputs on a small set of labeled lines; track CER/WER deltas after cleanup and correction.

## Optional recognizers and experiments
1) **Tesseract (tur, tur_best, tur_fast)**: good for printed Turkish; weak on handwriting. Use selectively on printed parts or numbers via `pytesseract.image_to_string(img, lang="tur")`.
2) **DocTR (Mindee)**: solid for Latin printed text; limited handwriting support. Use as a baseline on whole-page crops or clean line crops.
3) **MMOCR (OpenMMLab)**: many detectors/recognizers (DB, PSENet, CRAFT, CRNN, SAR). Heavy; best for research or fine-tuning on Turkish data.
4) **Keras-OCR**: CRAFT detector + CRNN recognizer. Simple to try; expect weaker performance on messy handwriting unless retrained.
5) **Handwriting-specific frameworks**: Kraken, Calamari, PyLaia for custom training if you collect 100–200+ labeled Turkish lines.
6) **Cloud/commercial**: Google Vision/Document AI, Azure Vision Read, ABBYY. Strong handwriting support; use via API if allowed.

## Stretch goals
- Collect a small labeled set of your own handwriting lines and fine-tune a handwriting model (TrOCR/CRNN) for Turkish.
- Benchmark multiple detectors (PaddleOCR DB vs. CRAFT) and see if better localization helps recognition.
- Automate a comparison script that runs all enabled recognizers on saved line crops and reports per-line metrics.
