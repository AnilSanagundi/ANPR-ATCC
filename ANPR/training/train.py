import os
import glob
import shutil
import cv2
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import random

# ------------------------
# PATHS
# ------------------------
DATASET = "dataset"
IMG_DIR = os.path.join(DATASET, "images")
ANN_DIR = os.path.join(DATASET, "labels")

TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/val"

# ------------------------
# TRAIN/VAL SPLIT
# ------------------------
# collect images recursively for common image extensions
img_patterns = [os.path.join(IMG_DIR, "**", "*.jpg"),
                os.path.join(IMG_DIR, "**", "*.jpeg"),
                os.path.join(IMG_DIR, "**", "*.png")]
images = []
for p in img_patterns:
    images.extend(glob.glob(p, recursive=True))

images = sorted(list(set(images)))
random.seed(42)
random.shuffle(images)

split = int(0.8 * len(images))
train_images = images[:split]
val_images   = images[split:]

for folder in [TRAIN_DIR, VAL_DIR]:
    for sub in ["images", "annotations", "labels"]:
        os.makedirs(os.path.join(folder, sub), exist_ok=True)

def move_files(image_list, dest):
    for img in image_list:
        name = os.path.basename(img)
        base, ext = os.path.splitext(name)
        shutil.copy(img, os.path.join(dest, "images", name))

        # prefer existing YOLO txt labels, otherwise xml annotations
        txt_name = base + ".txt"
        xml_name = base + ".xml"

        txt_src = os.path.join(ANN_DIR, txt_name)
        xml_src = os.path.join(ANN_DIR, xml_name)

        if os.path.exists(txt_src):
            shutil.copy(txt_src, os.path.join(dest, "labels", txt_name))
        elif os.path.exists(xml_src):
            shutil.copy(xml_src, os.path.join(dest, "annotations", xml_name))
        else:
            # no annotation found; create empty label file (allowed)
            open(os.path.join(dest, "labels", txt_name), "w").close()

move_files(train_images, TRAIN_DIR)
move_files(val_images, VAL_DIR)

print("✔ Train/Val split complete")

# ------------------------
# XML → YOLO
# ------------------------
def convert_xml(xml_file, w, h):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels = []

    for obj in root.findall("object"):
        box = obj.find("bndbox")
        xmin = int(box.find("xmin").text)
        ymin = int(box.find("ymin").text)
        xmax = int(box.find("xmax").text)
        ymax = int(box.find("ymax").text)

        x_center = (xmin + xmax) / 2 / w
        y_center = (ymin + ymax) / 2 / h
        width  = (xmax - xmin) / w
        height = (ymax - ymin) / h

        labels.append(f"0 {x_center} {y_center} {width} {height}")

    return labels


def process_folder(folder):
    img_dir = os.path.join(folder, "images")
    ann_dir = os.path.join(folder, "annotations")
    label_dir = os.path.join(folder, "labels")

    for img_name in os.listdir(img_dir):
        base, ext = os.path.splitext(img_name)
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, base + ".txt")

        # if a label txt already exists (copied from ANN_DIR), skip conversion
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            continue

        # try xml annotation conversion if xml present
        xml_path = os.path.join(ann_dir, base + ".xml")

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read image {img_path}")
            continue

        h, w, _ = img.shape

        if os.path.exists(xml_path):
            labels = convert_xml(xml_path, w, h)
            with open(label_path, "w") as f:
                f.write("\n".join(labels))
        else:
            # create empty label file if nothing found
            open(label_path, "w").close()

process_folder(TRAIN_DIR)
process_folder(VAL_DIR)

print("✔ XML → YOLO conversion done")

# ------------------------
# CREATE data.yaml
# ------------------------
yaml_text = """train: dataset/train/images
val: dataset/val/images

nc: 1
names: ["number_plate"]
"""

with open("data.yaml", "w") as f:
    f.write(yaml_text)

print("✔ data.yaml created")

# ------------------------
# TRAIN YOLOv8 MODEL
# ------------------------
print("🚀 Training started...")

model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)

print("🎉 Training Completed! Find best.pt inside: runs/detect/train/weights/")
