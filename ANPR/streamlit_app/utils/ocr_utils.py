import easyocr

# Initialize OCR model once
reader = easyocr.Reader(['en'], gpu=False)

def read_text(image):
    """
    Reads text from a processed plate image.
    Returns the detected text, or empty string if not found.
    """
    result = reader.readtext(image)
    if result:
        return result[0][1]   # Extract text
    return ""                 # If OCR fails
