"""
generate_sample_data.py — Generate realistic sample steel strip images & annotations
=====================================================================================
If you haven't downloaded the 13GB Kaggle Severstal dataset yet, this script creates
a realistic sample dataset in `data/` so you can train, test, and run the complete
pipeline immediately!

Defect types modeled (matching Severstal Steel Defect Detection):
  - Class 1: Pitted surface / small patches (clusters of dark indentations)
  - Class 2: Inclusions / micro-cracks (thin vertical or horizontal cracks)
  - Class 3: Scratches / gouges (long horizontal abrasive lines from rollers)
  - Class 4: Spots / oxidation scale (larger circular or irregular blotches)

Run with:
    python src/generate_sample_data.py
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import mask_to_rle

HEIGHT = 256
WIDTH = 1600


def create_steel_texture(height=HEIGHT, width=WIDTH, rng=None):
    """
    Generate realistic rolled stainless steel texture:
    - Base metallic gray tone (180-210)
    - Horizontal rolling striations / brushed lines
    - Subtle Gaussian noise and lighting gradient across strip
    """
    if rng is None:
        rng = np.random.default_rng()

    # Base metallic gray
    base_gray = rng.integers(170, 205)
    img = np.full((height, width), base_gray, dtype=np.float32)

    # Horizontal rolling grain (steel is rolled horizontally through rollers)
    striations = rng.normal(0, 5, size=(height, 1)).astype(np.float32)
    striations = cv2.resize(striations, (width, height))
    img += striations

    # High frequency surface roughness / speckles
    grain = rng.normal(0, 3.5, size=(height, width)).astype(np.float32)
    img += grain

    # Soft rolling illumination gradient
    grad = np.linspace(-6, 6, width, dtype=np.float32)
    img += grad

    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def add_defect_class_1(image, rng):
    """Class 1: Pitted surface / small patchy indentations."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    num_pits = rng.integers(15, 40)
    cx = rng.integers(100, WIDTH - 200)
    cy = rng.integers(30, HEIGHT - 50)
    radius_spread = rng.integers(30, 80)

    for _ in range(num_pits):
        px = int(np.clip(rng.normal(cx, radius_spread), 5, WIDTH - 6))
        py = int(np.clip(rng.normal(cy, radius_spread // 2), 5, HEIGHT - 6))
        pr = int(rng.integers(2, 5))
        cv2.circle(mask, (px, py), pr, 1, -1)
        # Darken defect pixels on image
        cv2.circle(image, (px, py), pr, (int(rng.integers(50, 90)), int(rng.integers(50, 90)), int(rng.integers(50, 90))), -1)

    return image, mask


def add_defect_class_2(image, rng):
    """Class 2: Inclusions / micro-cracks (thin irregular jagged fissures)."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    x0 = rng.integers(100, WIDTH - 250)
    y0 = rng.integers(40, HEIGHT - 60)
    length = rng.integers(60, 160)

    points = [(x0, y0)]
    curr_x, curr_y = x0, y0
    for _ in range(length // 8):
        curr_x += rng.integers(6, 12)
        curr_y += rng.integers(-3, 4)
        curr_y = int(np.clip(curr_y, 5, HEIGHT - 6))
        points.append((curr_x, curr_y))

    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(mask, [pts], False, 1, thickness=2)
    # Dark sharp crack on steel
    cv2.polylines(image, [pts], False, (40, 40, 45), thickness=2)

    return image, mask


def add_defect_class_3(image, rng):
    """Class 3: Scratches / gouges (long horizontal abrasive marks)."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    y = int(rng.integers(30, HEIGHT - 30))
    x_start = int(rng.integers(50, WIDTH - 450))
    length = int(rng.integers(250, 600))
    x_end = min(x_start + length, WIDTH - 10)
    thickness = int(rng.integers(2, 5))

    cv2.line(mask, (x_start, y), (x_end, y + int(rng.integers(-4, 5))), 1, thickness)
    # Highlight with dark line + slight bright reflection edge
    cv2.line(image, (x_start, y), (x_end, y), (60, 60, 65), thickness)
    cv2.line(image, (x_start, y + 1), (x_end, y + 1), (230, 230, 235), 1)

    return image, mask


def add_defect_class_4(image, rng):
    """Class 4: Spots / oxide scale (larger blotches with textured edges)."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cx = int(rng.integers(100, WIDTH - 200))
    cy = int(rng.integers(40, HEIGHT - 50))
    axes = (int(rng.integers(15, 35)), int(rng.integers(10, 25)))
    angle = int(rng.integers(0, 180))

    cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 1, -1)
    # Oxide patch has mottled dark/brownish appearance
    cv2.ellipse(image, (cx, cy), axes, angle, 0, 360, (75, 70, 65), -1)

    return image, mask


def generate_dataset(num_samples: int = 120, output_dir: str = "data"):
    """
    Generate sample images and matching train.csv with Severstal format.
    """
    rng = np.random.default_rng(42)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_data_dir = os.path.join(project_root, output_dir)
    images_dir = os.path.join(target_data_dir, "train_images")
    os.makedirs(images_dir, exist_ok=True)

    csv_rows = []
    print(f"Generating {num_samples} realistic steel strip samples...")

    defect_generators = {
        1: add_defect_class_1,
        2: add_defect_class_2,
        3: add_defect_class_3,
        4: add_defect_class_4,
    }

    # Class probabilities reflecting natural steel defect occurrence (Class 3 & 4 more common)
    for i in range(num_samples):
        img_id = f"sample_{i:05d}.jpg"
        img_path = os.path.join(images_dir, img_id)
        image = create_steel_texture(HEIGHT, WIDTH, rng=rng)

        # Decide which defects to place on this image:
        # ~30% clean images, ~50% single defect, ~20% multiple defects
        rand_val = rng.random()
        active_classes = []
        if rand_val > 0.30:
            if rand_val < 0.80:
                # 1 defect (weighted towards class 3 & 1)
                c = rng.choice([1, 2, 3, 4], p=[0.25, 0.15, 0.40, 0.20])
                active_classes.append(c)
            else:
                # 2 defects
                c1 = rng.choice([1, 3])
                c2 = rng.choice([2, 4])
                active_classes = [c1, c2]

        masks_by_class = {}
        for c in active_classes:
            image, mask = defect_generators[c](image, rng)
            masks_by_class[c] = mask

        # Save image
        cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Add 4 rows per image to match Severstal competition CSV format:
        # ImageId_ClassId, EncodedPixels
        for c in range(1, 5):
            row_id = f"{img_id}_{c}"
            if c in masks_by_class and masks_by_class[c].sum() > 0:
                rle = mask_to_rle(masks_by_class[c])
            else:
                rle = np.nan
            csv_rows.append({"ImageId_ClassId": row_id, "EncodedPixels": rle})

        if (i + 1) % 20 == 0 or (i + 1) == num_samples:
            print(f"  Processed {i+1}/{num_samples} images...")

    csv_df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(target_data_dir, "train.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"\n✓ Generated {num_samples} sample images in: {images_dir}")
    print(f"✓ Created Severstal-format CSV: {csv_path}")
    return csv_path, images_dir


if __name__ == "__main__":
    generate_dataset()
