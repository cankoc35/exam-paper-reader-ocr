from paddleocr import PaddleOCR
from segmentation import cropped_image

def main():
    ocr = PaddleOCR(
        lang="tr",               
        ocr_version="PP-OCRv5",  
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )

    results = ocr.predict(cropped_image)
    
    print("\n--- DETECTED LINES ---")
    for page in results:
        data = page.json["res"]
        rec_texts = data["rec_texts"]
        rec_scores = data["rec_scores"]

        for text, score in zip(rec_texts, rec_scores):
            print(f"{text!r}  (conf={score:.2f})")
    print("----------------------")

if __name__ == "__main__":
    main()


