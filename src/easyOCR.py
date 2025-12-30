import easyocr
import cv2
import numpy as np

# Initialize once to avoid reloading models per crop.
_READER = easyocr.Reader(["en"], gpu=False)

def run_easyocr(image):
    """
    Run EasyOCR on a cropped image (np.ndarray) or image path.
    Returns text strings only (no confidence).
    """
    if isinstance(image, np.ndarray):
        # EasyOCR expects RGB; OpenCV crops are BGR.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return _READER.readtext(rgb, detail=0)
    return _READER.readtext(str(image), detail=0)
    
