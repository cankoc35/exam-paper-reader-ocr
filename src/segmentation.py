from pathlib import Path
import cv2
import numpy as np

base_dir = Path(__file__).resolve().parent.parent
img_path = str(base_dir / "data" / "example_exam_03.jpeg")

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(img_path)

# 1) grayscale + blur
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# 2) global threshold (Otsu) – white page becomes 255
_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3) small close to remove tiny holes
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

# 4) find external contours in thresholded image
contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

h, w = img.shape[:2]
min_area = 0.25 * h * w
best = None

for c in sorted(contours, key=cv2.contourArea, reverse=True):
    area = cv2.contourArea(c)
    if area < min_area:
        continue

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # keep “big-ish” polygons, no hard aspect ratio:
    if len(approx) < 4:
        continue

    best = approx
    break

cropped_image = None
if best is None:
    print("No page contour found")
else:
    x, y, w, h = cv2.boundingRect(best)
    cropped_image = img[y:y + h, x:x + w]  # this is the BGR crop you pass to OCR
    dbg = img.copy()
    cv2.drawContours(dbg, [best], -1, (0, 255, 0), 3)
    cv2.namedWindow("page", cv2.WINDOW_NORMAL)
    cv2.imshow("page", dbg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
