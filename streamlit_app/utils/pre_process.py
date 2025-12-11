import cv2
import numpy as np

def preprocess_plate(crop):
    """
    Preprocesses the cropped number plate for better OCR accuracy.
    Steps:
    - Convert to grayscale
    - Sharpen image
    - Resize (2x)
    - Threshold
    """
    # Convert to gray
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Sharpen image
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    sharp = cv2.addWeighted(gray, 2.0, blur, -1.0, 0)

    # Resize for better OCR
    resized = cv2.resize(sharp, None, fx=2, fy=2)

    # Thresholding
    _, thresh = cv2.threshold(resized, 130, 255, cv2.THRESH_BINARY)

    return thresh

