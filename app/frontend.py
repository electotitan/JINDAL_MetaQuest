"""
frontend.py — Step 8: Streamlit Frontend for Defect Detection
==============================================================
A simple web UI where you:
  1. Upload a steel strip image
  2. Click "Detect Defects"
  3. See the results — defect type, location, and confidence

WHAT IS STREAMLIT?
  Streamlit lets you build web apps using plain Python — no HTML, CSS,
  or JavaScript needed. You just write Python code, and Streamlit turns
  it into a web page with buttons, file uploaders, and displays.

Run with:
    streamlit run app/frontend.py
    Then open http://localhost:8501 in your browser!

IMPORTANT: The FastAPI backend must be running first:
    uvicorn app.backend:app --host 0.0.0.0 --port 8000
"""

import streamlit as st
import requests
import base64
import io
from PIL import Image
import json

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Steel Defect Detector",
    page_icon="🔍",
    layout="wide",
)

# ============================================================
# App Header
# ============================================================
st.title("🔍 Steel Surface Defect Detector")
st.markdown("""
Upload a steel strip image to automatically detect and classify surface defects.
The AI model identifies **4 types of defects** and shows you exactly where they are.

**How it works:**
1. Upload an image (JPEG or PNG)
2. The AI model (U-Net with ResNet34) analyzes every pixel
3. You get back: defect type, location, and confidence score
""")

st.divider()

# ============================================================
# Sidebar Settings
# ============================================================
st.sidebar.header("⚙️ Settings")

# API URL (where the FastAPI backend is running)
api_url = st.sidebar.text_input(
    "Backend API URL",
    value="http://localhost:8000",
    help="The URL where the FastAPI backend is running",
)

# Confidence threshold slider
threshold = st.sidebar.slider(
    "Detection Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Higher = fewer detections but more confident. "
         "Lower = catches more defects but may have false alarms.",
)

# Color legend
st.sidebar.header("🎨 Defect Colors")
st.sidebar.markdown("""
- 🔴 **Class 1**: Patches
- 🟢 **Class 2**: Inclusions
- 🔵 **Class 3**: Scratches
- 🟡 **Class 4**: Spots
""")

# ============================================================
# Main Content
# ============================================================

# File uploader
uploaded_file = st.file_uploader(
    "Upload a steel strip image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Drag and drop or click to upload a steel strip image",
)

if uploaded_file is not None:
    # Show the uploaded image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        st.caption(f"Size: {image.size[0]} × {image.size[1]} pixels")

    # Detect button
    if st.button("🔍 Detect Defects", type="primary", use_container_width=True):
        with st.spinner("Analyzing image... (this takes a few seconds)"):
            try:
                # Send image to the FastAPI backend
                uploaded_file.seek(0)  # Reset file pointer
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                params = {"threshold": threshold}

                response = requests.post(
                    f"{api_url}/predict",
                    files=files,
                    params=params,
                    timeout=30,
                )

                if response.status_code != 200:
                    st.error(f"API Error: {response.status_code} — {response.text}")
                    st.stop()

                result = response.json()

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to the backend API!\n\n"
                    "Make sure the FastAPI server is running:\n"
                    "```\n"
                    "uvicorn app.backend:app --host 0.0.0.0 --port 8000\n"
                    "```"
                )
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        # ============================================================
        # Display Results
        # ============================================================

        with col2:
            st.subheader("🎯 Detection Results")

            # Show annotated image
            if result.get("annotated_image"):
                annotated_bytes = base64.b64decode(result["annotated_image"])
                annotated_image = Image.open(io.BytesIO(annotated_bytes))
                st.image(annotated_image, use_container_width=True)

        # Summary
        st.divider()
        st.subheader("📋 Summary")
        st.info(result.get("summary", "No results"))

        # Detailed results
        defects = result.get("defects", [])
        if defects:
            st.subheader("🔬 Detailed Findings")

            for defect in defects:
                confidence = defect["confidence"]
                # Color the confidence bar based on severity
                if confidence > 0.8:
                    bar_color = "red"
                elif confidence > 0.5:
                    bar_color = "orange"
                else:
                    bar_color = "yellow"

                with st.expander(
                    f"**{defect['class_name']}** — Confidence: {confidence:.1%}",
                    expanded=True,
                ):
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Confidence", f"{confidence:.1%}")
                    col_b.metric("Defect Area", f"{defect['defect_area_pixels']:,} px")

                    bbox = defect.get("bounding_box")
                    if bbox:
                        col_c.metric(
                            "Bounding Box",
                            f"{bbox['width']}×{bbox['height']}",
                        )
                        st.caption(
                            f"Location: x={bbox['x']}, y={bbox['y']}, "
                            f"width={bbox['width']}, height={bbox['height']}"
                        )

                    st.progress(confidence)
        else:
            st.success("✅ No defects detected! This steel strip appears to be clean.")

        # Raw JSON (collapsible)
        with st.expander("📦 Raw API Response (for debugging)"):
            # Remove the base64 image to keep the display clean
            display_result = {k: v for k, v in result.items() if k != "annotated_image"}
            st.json(display_result)

else:
    # Show instructions when no file is uploaded
    st.info(
        "👆 Upload a steel strip image above to get started!\n\n"
        "**Tip:** You can find sample images in the `data/train_images/` folder."
    )

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption(
    "Steel Surface Defect Detection — Built with PyTorch, U-Net, FastAPI & Streamlit | "
    "Dataset: Severstal Steel Defect Detection (Kaggle)"
)
