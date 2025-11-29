from segmentation import cropped_image
from paddleOCR import get_paddle_items
from trOCR import predict_trocr
from easyOCR import predict_easyocr

import numpy as np

def main(image):
    if image is None:
        print("No cropped image to process for OCR.")
        return
    
    items = get_paddle_items(image)  # Paddle detection + text

    # predictions = []
    # for item in items:
    #     poly = np.array(item["poly"])
    #     xs = poly[:, 0]
    #     ys = poly[:, 1]
    #     x_min, x_max = int(xs.min()), int(xs.max())
    #     y_min, y_max = int(ys.min()), int(ys.max())

    #     crop = image[y_min:y_max, x_min:x_max]

    #     trocr_text = predict_trocr(crop)
    #     easy_text  = predict_easyocr(crop)
        
    #     predictions.append({
    #         "paddle_text": item["text"],
    #         "paddle_score": item["score"],
    #         "trocr_text": trocr_text,
    #         "easy_text": easy_text,
    #     })
        
    for i, item in enumerate(items):
        poly = np.array(item["poly"])
        xs = poly[:, 0]
        ys = poly[:, 1]
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        crop = image[y_min:y_max, x_min:x_max]

        trocr_text = predict_trocr(crop)
        easy_text  = predict_easyocr(crop)

        print(f"\nLINE {i}")
        print(f"  PaddleOCR : {item['text']!r} (conf={item['score']:.2f})")
        print(f"  TrOCR     : {trocr_text!r}")
        print(f"  EasyOCR   : {easy_text!r}")

if __name__ == "__main__":
    main(cropped_image)
