from paddleocr import PaddleOCR

paddle_ocr = PaddleOCR(
    lang="tr",
    ocr_version="PP-OCRv5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
)

def get_paddle_items(image_bgr):
    pages = paddle_ocr.predict(image_bgr)
    page = pages[0]

    data = page.json["res"]
    polys  = data["rec_polys"]   # (N, 4, 2)
    texts  = data["rec_texts"]   # list[str]
    scores = data["rec_scores"]  # list[float]

    paddleOCR_items = []
    
    for poly, text, score in zip(polys, texts, scores):
        paddleOCR_items.append({
            "poly": poly,
            "text": text,
            "score": float(score),
        })
    
    return paddleOCR_items
