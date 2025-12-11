import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import altair as alt

from utils.detector import detect_plates
from utils.ocr_utils import read_text
from utils.pre_process import preprocess_plate

# Streamlit settings
st.set_page_config(page_title="ANPR System", page_icon="🚗", layout="wide")

st.title("🚗 Automatic Number Plate Recognition (ANPR)")
st.write("Upload an image to detect and read vehicle number plates.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# When file uploaded
if uploaded_file:
    # Save file
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Read image
    image = cv2.imread(file_path)
    st.image(image, channels="BGR", caption="Uploaded Image", width=600)

    # Detect plates
    st.subheader("🔍 Detecting Number Plates...")
    detection_results = detect_plates(image)

    results = []

    if detection_results:
        st.success(f"Detected {len(detection_results)} plate(s)")
        st.subheader("📌 Recognized Plates:")

        # Display detections with object information
        col1, col2 = st.columns(2)
        
        for i, detection in enumerate(detection_results):
            x1, y1, x2, y2, class_id, class_name, confidence = detection
            crop = image[y1:y2, x1:x2]

            processed = preprocess_plate(crop)
            text = read_text(processed)
            
            # Store results with object information
            results.append({
                "Image": uploaded_file.name,
                "Plate #": i + 1,
                "Number Plate": text,
                "Object Name": class_name,
                "Object ID": class_id,
                "Confidence": f"{confidence:.4f}"
            })

            # Display in columns
            if i % 2 == 0:
                col = col1
            else:
                col = col2
                
            with col:
                st.image(crop, channels="BGR", use_container_width=True)
                st.markdown(f"""
                    **Plate {i+1}**
                    - 📋 Number: `{text}`
                    - 🏷️ Object Name: `{class_name}`
                    - 🆔 Object ID: `{class_id}`
                    - 📊 Confidence: `{confidence:.2%}`
                """)

        # Create DataFrame with all information
        df = pd.DataFrame(results)
        df.to_csv("outputs/recognized.csv", index=False)

        st.subheader("📊 Detection Results Table")
        st.dataframe(df, use_container_width=True)

        st.subheader("📥 Download Results")
        with open("outputs/recognized.csv", "rb") as f:
            st.download_button(
                label="📥 Download CSV",
                data=f,
                file_name="recognized.csv",
                mime="text/csv",
            )

    else:
        st.error("❌ No number plate detected!")

# ---------------------------
# Data Visualization Section
# ---------------------------
st.markdown("---")
st.header("📈 Data Visualization")

csv_path = os.path.join("outputs", "recognized.csv")
if os.path.exists(csv_path):
    try:
        viz_df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        viz_df = pd.DataFrame()

    if not viz_df.empty:
        # Ensure Confidence is numeric
        if "Confidence" in viz_df.columns:
            viz_df["Confidence"] = pd.to_numeric(viz_df["Confidence"], errors="coerce")

        # Layout two columns for charts
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Detections by Object Name")
            if "Object Name" in viz_df.columns and not viz_df["Object Name"].isna().all():
                counts = viz_df["Object Name"].value_counts().reset_index()
                counts.columns = ["Object Name", "Count"]
                bar = alt.Chart(counts).mark_bar().encode(
                    x=alt.X("Object Name:N", sort="-y"),
                    y=alt.Y("Count:Q"),
                    tooltip=["Object Name", "Count"]
                )
                st.altair_chart(bar, use_container_width=True)
            else:
                st.info("No object name data available yet.")

            st.subheader("Top Recognized Number Plates")
            if "Number Plate" in viz_df.columns and not viz_df["Number Plate"].isna().all():
                top = viz_df["Number Plate"].value_counts().reset_index()
                top.columns = ["Number Plate", "Count"]
                st.table(top.head(10))
            else:
                st.info("No recognized plate text available yet.")

        with c2:
            st.subheader("Confidence Distribution")
            if "Confidence" in viz_df.columns and viz_df["Confidence"].notna().any():
                hist = alt.Chart(viz_df).mark_bar().encode(
                    alt.X("Confidence:Q", bin=alt.Bin(step=0.05, extent=[0,1])),
                    y="count()",
                    tooltip=[alt.Tooltip("count()", title="Detections")]
                )
                st.altair_chart(hist, use_container_width=True)
            else:
                st.info("No confidence scores available yet.")

            st.subheader("All Detection Records")
            st.dataframe(viz_df, use_container_width=True)

    else:
        st.info("No detection results yet. Run an image to populate `outputs/recognized.csv`.")
else:
    st.info("No detection CSV found. Detect plates to generate `outputs/recognized.csv`.")
