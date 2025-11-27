from pathlib import Path
import numpy as np
import cv2

def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

base_dir = Path(__file__).resolve().parent.parent
img_path = str(base_dir / "data" / "example_exam_01.jpeg")

original_image = cv2.imread(img_path)

# 1) Grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# 2) blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 3) edges detection 
edged_image = cv2.Canny(blurred_image, 30, 150)

# 3.5) close gaps in edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, kernel, iterations=2)

# 4) contours detection
contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5) pick a big 4-corner contour (the page)
h, w = original_image.shape[:2]
min_area = 0.5 * h * w          

page_contour = None

for c in sorted(contours, key=cv2.contourArea, reverse=True):
    area = cv2.contourArea(c)
    if area < min_area:
        continue  # too small, skip

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)

    if len(approx) == 4:        # we want a quadrilateral
        page_contour = approx
        break

if page_contour is None:
    print("No page contour found")
    exit()

# 6) draw the page contour
contour_img = original_image.copy()
cv2.drawContours(contour_img, [page_contour], -1, (0, 255, 0), 3)

# 7) perspective warp to get top-down page
rect = order_points(page_contour)
(tl, tr, br, bl) = rect

widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
maxWidth = int(max(widthA, widthB))

heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
maxHeight = int(max(heightA, heightB))

dst = np.array(
    [
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1],
    ],
    dtype="float32",
)

M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(original_image, M, (maxWidth, maxHeight))

warped_path = base_dir / "data" / "example_exam_01_warped.jpg"

# cv2.imwrite(str(warped_path), warped)
# print("Saved warped page to:", warped_path)

# cv2.imwrite(str(warped_gray_path), warped_gray)
# print("Saved warped gray page to:", warped_gray_path)

# 8) show both
cv2.namedWindow("page_contour", cv2.WINDOW_NORMAL)
cv2.imshow("page_contour", contour_img)

cv2.namedWindow("warped_page", cv2.WINDOW_NORMAL)
cv2.imshow("warped_page", warped)

cv2.waitKey(0)
cv2.destroyAllWindows()

