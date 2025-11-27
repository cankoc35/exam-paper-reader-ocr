from paddleocr import PaddleOCR
from segmentation import warped

def main():
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )

    results = ocr.predict(warped)
    
    print("\n--- DETECTED LINES ( WARPED COLOR ) ---")
    for page in results:
        data = page.json["res"]
        rec_texts = data["rec_texts"]
        rec_scores = data["rec_scores"]

        for text, score in zip(rec_texts, rec_scores):
            print(f"{text!r}  (conf={score:.2f})")
    print("----------------------")

if __name__ == "__main__":
    main()
