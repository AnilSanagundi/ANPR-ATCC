from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("models/best.pt")   # Make sure best.pt is inside /models/

def detect_plates(image):
    """
    Detects number plates in an image and returns their details.
    Returns a list of tuples: (x1, y1, x2, y2, class_id, class_name, confidence)
    """
    results = model(image)
    boxes = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            class_name = model.names[class_id]  # Class name from model
            
            boxes.append((x1, y1, x2, y2, class_id, class_name, conf))

    return boxes
