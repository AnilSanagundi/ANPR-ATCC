# 🚗 Automatic Number Plate Recognition (ANPR) – Complete Project

This project implements an **Automatic Number Plate Recognition (ANPR)** system using **YOLOv8**, **OpenCV**, and **OCR** to detect vehicle number plates and extract text from them.  
It includes a **Streamlit web app**, preprocessing utilities, a training module, and a results dashboard.

---

## 🧩 Project Overview

The system performs the following tasks:

1. **Detect vehicle number plates** using a YOLOv8 model  
2. **Crop and preprocess plates** (thresholding, noise removal, resizing)  
3. **Perform OCR** to extract the text  
4. **Store recognized outputs** in `outputs/recognized.csv`  
5. **Provide a simple UI** through Streamlit to upload images/videos  
6. **Allow retraining** using custom dataset under `training/`

---

## 📁 Folder Structure (As per your VS Code project)

Automatic Number Plate Recognition (ANPR) System
📌 Overview

The Automatic Number Plate Recognition (ANPR) system detects vehicle number plates from images or video frames and extracts the text using Optical Character Recognition (OCR).
This project combines Computer Vision, Deep Learning, and OCR to build a real-world traffic automation and security solution.

🧠 Features

📸 Automatic detection of number plates using YOLO/OpenCV.

🔠 Text extraction (OCR) using Tesseract/EasyOCR.

🧼 Pre-processing for improved accuracy (thresholding, deskewing, denoising).

📊 Streamlit web interface for user-friendly interaction.

📁 Upload image/video and view detection results.

💾 Saves recognized plate numbers with timestamps.

🚗 Works with Indian-style number plates (supports others too).

🛠️ Tech Stack
Backend / Processing

Python

OpenCV

YOLOv5 / Traditional contour-based detection

EasyOCR / Tesseract OCR

NumPy

Pandas

Frontend

Streamlit Web App

Optional Integrations

MySQL / MongoDB for storing vehicle logs

REST API for external apps

📂 Project Structure
ANPR/
│── models/
│   └── best.pt                 # YOLO model (if applicable)
│── utils/
│   ├── detector.py             # Number plate detection
│   ├── ocr_utils.py            # OCR logic
│   └── pre_process.py          # Image preprocessing
│── app.py                      # Streamlit main app
│── requirements.txt
│── README.md


2️⃣ Create Virtual Environment
python -m venv anpr_env


Activate it:

Windows
anpr_env\Scripts\activate
Linux/Mac
source anpr_env/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
streamlit run app.py

🧩 How ANPR Works (Pipeline)
🔹 Step 1: Image/Video Upload

User uploads vehicle image or video frame.

🔹 Step 2: Preprocessing

Convert to grayscale

Noise removal

Thresholding

Sharpening

🔹 Step 3: Detection

YOLO model locates plate region
OR

Edge detection + contour filtering

🔹 Step 4: OCR

Extract text using:

EasyOCR

or Tesseract

🔹 Step 5: Display & Save

Bounding box on plate

Extracted number

Timestamped log saved to CSV/Database

🌍 Real-World Applications

🚧 Smart Parking Systems

🛣️ Highway toll automation

🚦 Traffic rule violation detection

👮 Law enforcement (stolen vehicle tracking)

📸 Red-light & speed enforcement cameras

🏢 Access control for apartments/organizations

📊 Fleet & logistics monitoring

📦 Sample Output

Outputs include:
Detected number plate bounding box
Extracted text
Log entry with timestamp