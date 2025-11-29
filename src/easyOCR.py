import easyocr

easyOCR = easyocr.Reader(["tr", "en"], gpu=False)

def predict_easyocr(image):
    result = easyOCR.readtext(image, detail=0, paragraph=True)
    return result[0] if result else ""


