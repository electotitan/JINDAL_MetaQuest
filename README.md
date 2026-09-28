# 🔍 AI-Powered Steel Surface Defect Detection

> Detect, localize, and classify 4 types of surface defects on stainless steel strips using Deep Learning (U-Net with ResNet-34) and computer vision, accessible via a real-time web application.

---

## 📌 Project Overview

In industrial steel manufacturing, surface defects (such as pitting, inclusion cracks, rolling scratches, and oxidation spots) reduce strip strength, cause corrosion vulnerability, and result in costly quality rejection. This project provides an automated, high-precision computer vision pipeline built upon the **Severstal Steel Defect Detection** benchmark:

1. **Pixel-Level Segmentation**: Instead of a simple "yes/no" defect flag, a **U-Net** architecture outputs pixel-exact defect boundaries across 4 distinct classes.
2. **High Sensitivity (Recall ≥ 90%)**: Prioritizes capturing all true defects via a custom **Weighted Binary Cross-Entropy + Dice Loss**.
3. **Sub-100ms Inference**: Optimized for real-time factory line cameras.
4. **Interactive Web App**: End-to-end user interface with **FastAPI** backend and **Streamlit** frontend.

---

## 🧱 Defect Taxonomy

| Defect Class | Name & Industrial Cause | Visual Characteristic | Color Legend |
|---|---|---|---|
| **Class 1** | Pitted surface / indentations (chemical/mechanical pitting) | Dense cluster of dark surface pits | 🔴 **Red** |
| **Class 2** | Inclusions / micro-cracks (foreign particulates or fissures) | Thin, irregular sharp cracks | 🟢 **Green** |
| **Class 3** | Scratches / gouges (abrasion from steel rolling guides) | Long horizontal striation lines | 🔵 **Blue** |
| **Class 4** | Spots / oxidation scale (thermal oxidation or rolling scale) | Irregular oval blotches / patches | 🟡 **Yellow** |

---

## 📁 Project Structure

```
steel-defect-detection/
├── app/
│   ├── backend.py            # FastAPI REST server with /predict endpoint
│   └── frontend.py           # Streamlit user interface
├── data/
│   ├── train.csv             # ImageId_ClassId & RLE encoded masks
│   └── train_images/         # Raw steel strip images (256x1600 px)
├── models/
│   ├── best_model.pth        # Saved PyTorch checkpoint
│   ├── training_curves.png   # Loss, Dice, and IoU plots
│   └── predictions_vs_truth.png
├── notebooks/                # Exploratory notebooks
├── src/
│   ├── utils.py              # RLE encode/decode, mask overlays
│   ├── generate_sample_data.py # Sample data generator for quick test runs
│   ├── download_dataset.py   # Kaggle CLI downloader for full 13GB dataset
│   ├── explore_data.py       # Data analysis & class imbalance reporting
│   ├── dataset.py            # PyTorch Dataset, Albumentations, DataLoaders
│   ├── model.py              # U-Net + ResNet-34 encoder (ImageNet pretrained)
│   ├── train.py              # Training loop with mixed precision & early stopping
│   ├── evaluate.py           # Test set evaluation, Dice/IoU/Recall, 5-fold CV
│   └── optimize.py           # Latency benchmark (<100ms) & backbone comparison
├── Dockerfile                # Single-command container deployment
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Virtual Environment

```bash
cd steel-defect-detection
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Dataset

You can work with the dataset in two ways:

#### Option A: Quick Run with Sample Data (Instant)
```bash
python src/generate_sample_data.py
```
This generates realistic steel strip images (256x1600) and `train.csv` in `data/` immediately.

#### Option B: Full Kaggle Severstal Dataset (~13 GB)
1. Place your `kaggle.json` API token in `~/.kaggle/`
2. Run:
```bash
python src/download_dataset.py
```

---

## 🔬 Pipeline Execution Step-by-Step

### Step 2: Explore Data & Check Class Imbalance
```bash
python src/explore_data.py
```
- Decodes RLE masks.
- Prints exact defect distribution across the 4 classes.
- Saves `data/data_exploration.png` showing steel images with colored defect overlays.

### Step 3: Test DataLoader & Augmentations
```bash
python src/dataset.py
```
- Verifies stratified train/val/test split.
- Tests random horizontal/vertical flips, brightness adjustments, and normalization.

### Step 4: Verify Model Architecture
```bash
python src/model.py
```
- Instantiates U-Net with ResNet-34 backbone.
- Verifies input tensor shape `(B, 3, 256, 512)` → output tensor shape `(B, 4, 256, 512)`.

### Step 5: Train the Model
```bash
python src/train.py
```
- Runs training with Combined Loss (Weighted BCE + Dice Loss).
- Enables automatic mixed precision (`torch.cuda.amp`) for speed and memory efficiency.
- Early stopping halts training automatically if validation Dice doesn't improve for 7 epochs.
- Saves the best checkpoint to `models/best_model.pth`.

### Step 6: Evaluate on Held-Out Test Set
```bash
python src/evaluate.py
```
- Computes per-class **Dice Score**, **IoU**, **Recall**, and **False-Alarm Rate**.
- Generates side-by-side visualization: `Original Image` | `Ground Truth` | `Predicted Mask`.
- Includes optional 5-fold cross-validation.

### Step 7: Benchmark Inference Latency
```bash
python src/optimize.py
```
- Measures average inference latency per image on your GPU.
- Verifies < 100ms threshold for real-time camera inspection.
- Compares ResNet-34 vs. MobileNetV2 and EfficientNet backbones.

---

## 🌐 Running the Web Application

Launch the **FastAPI backend** and **Streamlit frontend**:

### Terminal 1: Backend API
```bash
uvicorn app.backend:app --host 0.0.0.0 --port 8000
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### Terminal 2: Streamlit Frontend
```bash
streamlit run app/frontend.py
```
Open your browser at `http://localhost:8501`:
1. Upload any steel strip image.
2. Adjust confidence threshold slider (default: `0.50`).
3. Click **"Detect Defects"** to see defect classification, bounding boxes, pixel areas, and color overlays.

---

## 🐳 Docker Deployment

To build and run both the API and Streamlit UI inside a single container:

```bash
docker build -t steel-defect-detector .
docker run -p 8000:8000 -p 8501:8501 steel-defect-detector
```

---

## 📚 Key ML Concepts Reference (Beginner Friendly)

- **RLE (Run-Length Encoding)**: A data compression format storing defect locations as `(start_pixel, length)` pairs instead of millions of 0s and 1s.
- **U-Net**: An encoder-decoder neural network shaped like a "U". The encoder extracts deep context, while skip connections pass fine spatial details directly to the decoder for pixel-precise defect boundaries.
- **Dice Coefficient**: Overlap metric between 0 and 1. Measures how closely the predicted defect polygon matches the real defect shape.
- **IoU (Intersection over Union)**: Stricter overlap metric: `Intersection / Union`.
- **Recall**: Proportion of actual defects detected (`True Positives / (True Positives + False Negatives)`). Target: ≥ 90%.
- **False-Alarm Rate**: Frequency of non-defective steel misidentified as defects (`False Positives / (False Positives + True Negatives)`).
- **Quantization**: Compressing weights from 32-bit floats to 8-bit integers for 2-4x faster inference with minimal accuracy drop.# JINDAL_MetaQuest
