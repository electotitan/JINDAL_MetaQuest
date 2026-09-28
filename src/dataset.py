"""
dataset.py — Step 3: PyTorch Dataset & DataLoader for steel defect detection
=============================================================================
This file handles:
  1. Splitting data into train / validation / test sets (stratified)
  2. A PyTorch Dataset class that loads images + masks on the fly
  3. Image augmentation (making training data more diverse)
  4. DataLoaders that feed batches of data to the model during training

KEY TERMS:
  - Dataset: A class that tells PyTorch how to load one image + its mask
  - DataLoader: Wraps a Dataset and feeds it to the model in batches
  - Augmentation: Randomly modifying images (flips, brightness changes) so the
    model sees more variety and doesn't just memorize the training images
  - Stratified split: When splitting data, we make sure each split has a similar
    proportion of each defect class (so no class is accidentally left out)
  - Normalization: Scaling pixel values to a standard range that the model expects
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import rle_to_mask


# ============================================================
# Image size and normalization settings
# ============================================================
# Original images are 256 x 1600 pixels — that's very wide!
# We resize to 256 x 512 to fit in GPU memory (4 GB VRAM).
# ImageNet mean/std are used because our encoder (ResNet34) was
# pretrained on ImageNet, so it expects inputs in this range.
IMG_HEIGHT = 256
IMG_WIDTH = 512

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # Average pixel values ImageNet uses
IMAGENET_STD = (0.229, 0.224, 0.225)   # How spread out the pixel values are

NUM_CLASSES = 4  # 4 types of defects


def get_augmentation_pipeline(phase: str) -> A.Compose:
    """
    Create an image augmentation pipeline.

    WHY AUGMENTATION?
    If we train with the exact same images every time, the model might
    "memorize" them instead of learning general patterns. By randomly
    flipping, changing brightness, etc., we create slightly different
    versions each time. This teaches the model to recognize defects
    regardless of orientation or lighting.

    Parameters
    ----------
    phase : str
        "train" → apply random augmentations
        "val" or "test" → only resize and normalize (no randomness!)
          We don't augment validation/test because we want consistent
          measurements of how good the model is.
    """
    if phase == "train":
        return A.Compose([
            # Resize to our target dimensions
            A.Resize(IMG_HEIGHT, IMG_WIDTH),

            # Randomly flip the image left-right (50% chance)
            # Defects can appear on either side, so this helps
            A.HorizontalFlip(p=0.5),

            # Randomly flip top-bottom (50% chance)
            A.VerticalFlip(p=0.5),

            # Randomly change brightness and contrast (30% chance)
            # Real factory lighting varies, so this simulates that
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3,
            ),

            # Randomly shift color channels slightly (20% chance)
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.2,
            ),

            # Add slight Gaussian noise (20% chance)
            # Real cameras have noise — this makes the model robust to it
            A.GaussNoise(p=0.2),

            # Normalize pixel values to what ImageNet-pretrained models expect
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

            # Convert from numpy array to PyTorch tensor
            ToTensorV2(),
        ])
    else:
        # Validation / test: just resize and normalize, no randomness
        return A.Compose([
            A.Resize(IMG_HEIGHT, IMG_WIDTH),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


class SteelDataset(Dataset):
    """
    PyTorch Dataset for the Severstal Steel Defect Detection dataset.

    WHAT IS A PYTORCH DATASET?
    It's a class that tells PyTorch: "Here's how to load one example."
    PyTorch will then call __getitem__ repeatedly to build batches.

    Each "example" is a pair: (image, masks)
      - image: the steel strip photo (3 color channels: Red, Green, Blue)
      - masks: 4 binary masks (one per defect class), each showing where
        that type of defect appears in the image
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        phase: str = "train",
    ):
        """
        Parameters
        ----------
        dataframe : pd.DataFrame
            Must have columns: ImageId, ClassId, EncodedPixels, HasDefect
        image_dir : str
            Path to the folder containing the images
        phase : str
            "train", "val", or "test" — controls whether augmentation is applied
        """
        self.image_dir = image_dir
        self.phase = phase
        self.augmentation = get_augmentation_pipeline(phase)

        # Get unique image IDs (each image appears up to 4 times in the CSV,
        # once per defect class)
        self.image_ids = dataframe["ImageId"].unique()

        # Build a lookup: for each image, store its defect masks
        # This is faster than searching the dataframe every time
        self.masks_dict = {}
        for image_id in self.image_ids:
            image_rows = dataframe[dataframe["ImageId"] == image_id]
            self.masks_dict[image_id] = image_rows

    def __len__(self):
        """How many images in this dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        """
        Load one image and its 4 defect masks.

        This is called by PyTorch's DataLoader to build batches.
        """
        image_id = self.image_ids[idx]

        # --- Load the image ---
        img_path = os.path.join(self.image_dir, image_id)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Build the 4 defect masks ---
        # Shape: (256, 1600) for each class, then we'll resize later
        h, w = image.shape[:2]  # Original image dimensions
        masks = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)

        image_rows = self.masks_dict[image_id]
        for _, row in image_rows.iterrows():
            class_id = row["ClassId"]
            if row["HasDefect"]:
                mask = rle_to_mask(row["EncodedPixels"], height=h, width=w)
                masks[:, :, class_id - 1] = mask  # ClassId is 1-indexed, array is 0-indexed

        # --- Apply augmentation ---
        # albumentations handles both image AND mask transforms together,
        # so flips/resizes are applied consistently to both
        augmented = self.augmentation(image=image, mask=masks)
        image = augmented["image"]       # Now a PyTorch tensor: (3, H, W)
        masks = augmented["mask"]        # Now: (H, W, 4)

        # Rearrange masks to (4, H, W) — PyTorch expects channels first
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)
        masks = masks.permute(2, 0, 1).float()  # (H, W, 4) → (4, H, W)

        return image, masks


def prepare_data_splits(
    csv_path: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    """
    Split the dataset into train / validation / test sets.

    STRATIFIED SPLIT means we make sure each split has a similar proportion
    of each defect class. Without this, we might accidentally put all the
    rare defects in one split, making the model unable to learn them.

    Parameters
    ----------
    csv_path : str
        Path to train.csv
    test_size : float
        Fraction of data for testing (0.15 = 15%)
    val_size : float
        Fraction of data for validation (0.15 = 15%)

    Returns
    -------
    train_df, val_df, test_df : DataFrames
        The three splits, each containing the relevant rows
    """
    # Load and parse the CSV
    df = pd.read_csv(csv_path)
    df["ImageId"] = df["ImageId_ClassId"].apply(lambda x: x.split("_")[0])
    df["ClassId"] = df["ImageId_ClassId"].apply(lambda x: int(x.split("_")[1]))
    df["HasDefect"] = df["EncodedPixels"].notna()

    # For stratification, we need one label per image.
    # We'll use the "most severe" defect class, or 0 if no defect.
    image_labels = df.groupby("ImageId").apply(
        lambda g: g.loc[g["HasDefect"], "ClassId"].max() if g["HasDefect"].any() else 0
    ).reset_index()
    image_labels.columns = ["ImageId", "StratifyLabel"]

    # First split: separate out the test set
    train_val_ids, test_ids = train_test_split(
        image_labels["ImageId"],
        test_size=test_size,
        stratify=image_labels["StratifyLabel"],
        random_state=random_state,
    )

    # Second split: separate train and validation from the remaining data
    train_val_labels = image_labels[image_labels["ImageId"].isin(train_val_ids)]
    adjusted_val_size = val_size / (1 - test_size)  # Adjust ratio for the remaining data

    train_ids, val_ids = train_test_split(
        train_val_labels["ImageId"],
        test_size=adjusted_val_size,
        stratify=train_val_labels["StratifyLabel"],
        random_state=random_state,
    )

    # Filter the full dataframe by the image IDs in each split
    train_df = df[df["ImageId"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["ImageId"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["ImageId"].isin(test_ids)].reset_index(drop=True)

    print(f"Data split complete:")
    print(f"  Train: {train_ids.nunique()} images")
    print(f"  Val:   {val_ids.nunique()} images")
    print(f"  Test:  {test_ids.nunique()} images")

    return train_df, val_df, test_df


def get_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Calculate weights for each defect class based on how common they are.

    WHY CLASS WEIGHTS?
    If Class 3 has 10x more examples than Class 2, the model will focus
    on Class 3 (because getting it right improves the overall score more).
    By giving rare classes higher weights, we tell the model:
    "Pay EXTRA attention to these rare defects — they matter more per example."

    The weight for each class is: total_samples / (num_classes * class_count)
    So rare classes get higher weights, common classes get lower weights.
    """
    class_counts = []
    for class_id in range(1, NUM_CLASSES + 1):
        count = train_df[
            (train_df["ClassId"] == class_id) & (train_df["HasDefect"])
        ].shape[0]
        # Avoid division by zero for classes with no examples
        class_counts.append(max(count, 1))

    total = sum(class_counts)
    weights = [total / (NUM_CLASSES * c) for c in class_counts]

    # Normalize so the average weight is 1.0
    avg = sum(weights) / len(weights)
    weights = [w / avg for w in weights]

    print(f"Class weights (higher = model pays more attention):")
    for i, w in enumerate(weights):
        print(f"  Class {i+1}: {w:.3f} (count: {class_counts[i]})")

    return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_dir: str,
    batch_size: int = 8,
    num_workers: int = 2,
):
    """
    Create PyTorch DataLoaders for train, val, and test sets.

    WHAT IS A DATALOADER?
    A DataLoader takes a Dataset and serves it up in "batches" — small
    groups of images (e.g., 8 at a time). This is more efficient than
    feeding images one by one, because the GPU can process a batch in
    parallel.

    Parameters
    ----------
    batch_size : int
        How many images per batch. Larger = faster training, but uses
        more GPU memory. With 4GB VRAM, 8 should work; reduce to 4 if
        you get out-of-memory errors.
    num_workers : int
        How many CPU threads load images in the background while the
        GPU processes the current batch. 2 is a safe default on Windows.
    """
    train_dataset = SteelDataset(train_df, image_dir, phase="train")
    val_dataset = SteelDataset(val_df, image_dir, phase="val")
    test_dataset = SteelDataset(test_df, image_dir, phase="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Randomize order each epoch — helps learning
        num_workers=num_workers,
        pin_memory=True,    # Speeds up GPU transfer on CUDA
        drop_last=True,     # Drop the last incomplete batch (avoids batch norm issues)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# Quick test — run this file directly to verify everything works
# ============================================================
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "train.csv")
    image_dir = os.path.join(project_root, "data", "train_images")

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Download the dataset first.")
        sys.exit(1)

    # Split the data
    train_df, val_df, test_df = prepare_data_splits(csv_path)

    # Calculate class weights
    weights = get_class_weights(train_df)

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, image_dir, batch_size=4
    )

    # Test loading one batch
    print(f"\nTesting: loading one batch from train_loader...")
    images, masks = next(iter(train_loader))
    print(f"  Image batch shape: {images.shape}")   # Expected: (4, 3, 256, 512)
    print(f"  Mask batch shape:  {masks.shape}")     # Expected: (4, 4, 256, 512)
    print(f"  Image dtype: {images.dtype}, range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"  Mask dtype:  {masks.dtype}, unique values: {masks.unique().tolist()}")
    print("\nDataset setup looks good!")
