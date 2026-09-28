"""
backend.py — Step 8: FastAPI Backend for Defect Detection
==========================================================
This creates a web API with one endpoint:
  POST /predict — Upload a steel image → Get defect predictions back

The API returns:
  - Which defect classes were detected
  - Confidence score for each class
  - A base64-encoded image with defect masks drawn on top
  - Bounding box coordinates for each detected defect

WHAT IS AN API?
  An API (Application Programming Interface) is like a waiter at a
  restaurant. You (the frontend) send a request ("I want predictions
  for this image"), the API takes it to the kitchen (the model), and
  brings back the results.

WHAT IS FastAPI?
  A modern Python framework for building APIs. It's fast, easy to use,
  and automatically generates documentation you can try in the browser.

Run with:
    uvicorn app.backend:app --host 0.0.0.0 --port 8000 --reload
    Then open http://localhost:8000/docs to test the API interactively!
"""

import os
import sys
import io
import base64
import numpy as np
import cv2
import torch
from torch.cuda.amp import autocast
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import create_model
from src.dataset import IMG_HEIGHT, IMG_WIDTH, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


# ============================================================
# Initialize the FastAPI app
# ============================================================
app = FastAPI(
    title="Steel Defect Detection API",
    description="Upload a steel strip image to detect surface defects. "
                "Returns defect classes, confidence scores, and annotated image.",
    version="1.0.0",
)

# Allow requests from any origin (needed for Streamlit to talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Global variables — loaded once when the server starts
# ============================================================
model = None
device = None
transform = None

# Defect class names and colors (RGB)
CLASS_NAMES = {
    1: "Defect Type 1 (Patches)",
    2: "Defect Type 2 (Inclusions)",
    3: "Defect Type 3 (Scratches)",
    4: "Defect Type 4 (Spots)",
}
CLASS_COLORS = {
    1: (255, 0, 0),       # Red
    2: (0, 255, 0),       # Green
    3: (0, 0, 255),       # Blue
    4: (255, 255, 0),     # Yellow
}


@app.on_event("startup")
def load_model():
    """
    Load the trained model when the server starts.
    This only happens once, so subsequent requests are fast.
    """
    global model, device, transform

    # Set up device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "best_model.pth")

    if not os.path.exists(model_path):
        print(f"WARNING: Model not found at {model_path}")
        print("The API will return an error for predictions.")
        print("Run 'python src/train.py' first to train the model.")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    encoder_name = checkpoint.get("encoder_name", "resnet34")
    model = create_model(encoder_name=encoder_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    print(f"  Encoder: {encoder_name}")
    print(f"  Val Dice at save: {checkpoint.get('val_dice', 'N/A')}")

    # Image preprocessing pipeline (same as test-time transforms)
    transform = A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_bounding_box(mask: np.ndarray) -> dict:
    """
    Find the bounding box around a binary mask.
    Returns {x, y, width, height} or None if no defect pixels.
    """
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    y_min, y_max = int(coords[0].min()), int(coords[0].max())
    x_min, x_max = int(coords[1].min()), int(coords[1].max())
    return {
        "x": x_min,
        "y": y_min,
        "width": x_max - x_min + 1,
        "height": y_max - y_min + 1,
    }


def create_annotated_image(
    original_image: np.ndarray,
    pred_masks: np.ndarray,
    confidences: dict,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Draw defect masks and labels on the original image.
    Returns the annotated image (RGB).
    """
    annotated = original_image.copy()

    for class_id in range(1, NUM_CLASSES + 1):
        mask = pred_masks[class_id - 1]
        if mask.sum() == 0:
            continue

        color = CLASS_COLORS[class_id]
        # Draw semi-transparent mask overlay
        for c in range(3):
            annotated[:, :, c] = np.where(
                mask > 0,
                annotated[:, :, c] * 0.6 + color[c] * 0.4,
                annotated[:, :, c],
            )

        # Draw bounding box
        bbox = get_bounding_box(mask)
        if bbox:
            cv2.rectangle(
                annotated,
                (bbox["x"], bbox["y"]),
                (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
                color, 2,
            )
            # Add label
            label = f"Class {class_id}: {confidences.get(class_id, 0):.1%}"
            cv2.putText(
                annotated, label,
                (bbox["x"], max(bbox["y"] - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

    return annotated.astype(np.uint8)


@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    """
    Upload a steel strip image and get defect predictions.

    Parameters
    ----------
    file : UploadFile
        The image file (JPEG or PNG)
    threshold : float
        Confidence threshold (0-1). Pixels with confidence above this
        are classified as defects. Default 0.5.

    Returns
    -------
    JSON with:
      - defects: list of detected defects with class, confidence, bounding box
      - annotated_image: base64-encoded PNG with defects highlighted
      - summary: human-readable summary
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first with 'python src/train.py'"
        )

    # Read and validate the uploaded image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    original_h, original_w = image_np.shape[:2]

    # Preprocess the image (resize, normalize, convert to tensor)
    augmented = transform(image=image_np)
    input_tensor = augmented["image"].unsqueeze(0).to(device)  # Add batch dimension

    # Run the model
    with torch.no_grad():
        with autocast(device_type=device.type):
            output = model(input_tensor)

    # Convert output to probabilities using sigmoid
    probabilities = torch.sigmoid(output).cpu().numpy()[0]  # Shape: (4, H, W)

    # Create binary masks and compute per-class confidence
    defects = []
    confidences = {}
    pred_masks = np.zeros((NUM_CLASSES, original_h, original_w), dtype=np.uint8)

    for class_id in range(1, NUM_CLASSES + 1):
        prob_map = probabilities[class_id - 1]  # (H, W) probability map

        # Resize probability map back to original image size
        prob_map_resized = cv2.resize(prob_map, (original_w, original_h))

        # Apply threshold to get binary mask
        binary_mask = (prob_map_resized > threshold).astype(np.uint8)
        pred_masks[class_id - 1] = binary_mask

        if binary_mask.sum() > 0:
            # Confidence = average probability in the defect region
            confidence = float(prob_map_resized[binary_mask == 1].mean())
            confidences[class_id] = confidence
            bbox = get_bounding_box(binary_mask)

            defects.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "confidence": round(confidence, 4),
                "bounding_box": bbox,
                "defect_area_pixels": int(binary_mask.sum()),
            })

    # Create annotated image
    annotated = create_annotated_image(image_np, pred_masks, confidences, threshold)

    # Encode annotated image as base64 PNG
    _, buffer = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    # Build summary
    if defects:
        summary = f"Found {len(defects)} defect(s): " + ", ".join(
            [f"{d['class_name']} ({d['confidence']:.1%})" for d in defects]
        )
    else:
        summary = "No defects detected — this steel strip looks clean!"

    return JSONResponse({
        "defects": defects,
        "annotated_image": annotated_b64,
        "summary": summary,
        "image_size": {"width": original_w, "height": original_h},
        "threshold_used": threshold,
    })


@app.get("/")
def root():
    """Health check / welcome endpoint."""
    return {
        "message": "Steel Defect Detection API is running!",
        "docs": "Visit /docs for interactive API documentation",
        "predict": "POST /predict with an image file to get predictions",
    }


@app.get("/health")
def health():
    """Health check — used by Docker/monitoring to verify the service is alive."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized",
    }
