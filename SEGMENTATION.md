## Paper Extraction Pipeline (Professional & Robust)

### 1. Load image

* Read image with OpenCV
* Immediately verify it loaded correctly

### 2. Resize (performance + stability)

* Resize while keeping aspect ratio (e.g. max width 1200–1600 px)
* Store scale factor for later coordinate mapping

### 3. Illumination normalization (poor light handling)

* Convert to grayscale
* Apply CLAHE to improve local contrast
* If strong shadows exist: estimate background illumination using a large Gaussian blur and divide/subtract
* If color cast exists (yellow/blue light): apply simple white-balance (gray-world)

### 4. Noise reduction

* Apply light Gaussian or bilateral filtering
* Avoid aggressive blur (text must remain sharp)

### 5. Paper–background separation (mask creation)

**Option A – Edge-based**

* Canny edge detection
* Morphological dilation + closing to connect broken paper edges

**Option B – Region-based (recommended for poor light)**

* Adaptive thresholding or Otsu thresholding
* Keep the largest bright connected component

### 6. Main paper candidate selection

* Find contours from the mask
* Score candidates by:

  * Largest area
  * Rectangularity
  * Proximity to image center

### 7. Quadrilateral detection (ideal case)

* Approximate contour using `approxPolyDP`
* If exactly 4 points are found:

  * Order points
  * Apply perspective transform

### 8. Corner-missing fallback strategies (critical)

**Option A – Minimum area rectangle**

* Use `minAreaRect` on the contour
* Convert to 4 box points
* Apply perspective warp

**Option B – Line-based estimation**

* Detect dominant lines using Hough transform
* Extend and intersect lines to estimate page rectangle

### 9. Quality checks (automatic)

* Detect under/over-exposure (mean intensity too low/high)
* Detect strong shadow gradients
* If failed: re-run steps 3–5 with adjusted parameters

### 10. Validation

* Check aspect ratio is paper-like (A4 / Letter)
* Ensure borders exist (paper should not fill entire frame)
* If invalid: retry with alternative fallback strategy

### 11. Final crop and padding

* Add small margin before warping to avoid cutting edges

### 12. Output

* Save warped, paper-only image for OCR
* Preserve transform matrix if OCR boxes must be mapped back to original image
