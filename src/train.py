"""
train.py — Step 5: Training the U-Net model
=============================================
This script handles the complete training pipeline:
  1. Load data and create DataLoaders
  2. Build the U-Net model
  3. Train with combined BCE + Dice loss
  4. Use early stopping to avoid overfitting
  5. Log metrics and plot training curves
  6. Save the best model

KEY TERMS EXPLAINED:
  - EPOCH: One complete pass through ALL training images. If we have 8,000
    training images and batch size 8, one epoch = 1,000 batches.
  - LOSS: A number that measures "how wrong the model is." Lower = better.
    The model adjusts itself to make this number go down.
  - LEARNING RATE: How big of a step the model takes when adjusting itself.
    Too big → it overshoots and gets worse. Too small → it learns too slowly.
  - OVERFITTING: When the model memorizes the training images instead of
    learning general patterns. It gets great on training data but terrible
    on new images. Early stopping prevents this.
  - EARLY STOPPING: "If the model hasn't improved in N epochs, stop training."
    This saves time and prevents overfitting.

Run with:
    python src/train.py
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import prepare_data_splits, create_dataloaders, get_class_weights
from src.model import create_model


# ============================================================
# Loss Functions
# ============================================================

class DiceLoss(nn.Module):
    """
    Dice Loss — measures how much the predicted mask overlaps with the truth.

    SIMPLE EXPLANATION:
    Imagine two circles drawn on paper (one is the true defect, one is
    what the model predicted). Dice measures how much they overlap:
      - 1.0 = perfect overlap (great!)
      - 0.0 = no overlap at all (terrible!)

    We use (1 - Dice) as a loss, so lower = better overlap.

    The "smooth" term prevents division by zero when there are no defects.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to convert raw model outputs to probabilities (0-1)
        predictions = torch.sigmoid(predictions)

        # Flatten to 1D for easier computation
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)

        # Dice = 2 * |A ∩ B| / (|A| + |B|)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined loss: Weighted BCE + Dice Loss.

    WHY TWO LOSSES?
      - BCE (Binary Cross-Entropy) checks EACH PIXEL independently:
        "Is this pixel a defect or not?" Good for pixel accuracy.
      - Dice Loss checks the OVERALL SHAPE: "Does the predicted mask
        match the true mask shape?" Good for getting boundaries right.
      - Together, they give both pixel-level accuracy AND shape matching.

    CLASS WEIGHTS:
      Each defect class gets a weight. Rare classes get higher weights,
      so the model pays more attention to them.
    """

    def __init__(self, class_weights: torch.Tensor = None, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()

        if class_weights is not None:
            self.class_weights = class_weights
        else:
            self.class_weights = None

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : tensor (batch, 4, H, W) — raw model output (logits)
        targets : tensor (batch, 4, H, W) — true binary masks
        """
        total_loss = 0.0
        num_classes = predictions.shape[1]

        for c in range(num_classes):
            pred_c = predictions[:, c, :, :]
            target_c = targets[:, c, :, :]

            # BCE loss for this class
            bce = nn.functional.binary_cross_entropy_with_logits(
                pred_c, target_c, reduction="mean"
            )

            # Dice loss for this class
            dice = self.dice_loss(pred_c.unsqueeze(1), target_c.unsqueeze(1))

            # Combine BCE and Dice
            class_loss = self.bce_weight * bce + (1 - self.bce_weight) * dice

            # Apply class weight (rare classes matter more)
            if self.class_weights is not None:
                weight = self.class_weights[c]
                class_loss = class_loss * weight

            total_loss += class_loss

        return total_loss / num_classes


# ============================================================
# Metrics
# ============================================================

def compute_dice_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute the Dice coefficient (0 to 1, higher is better).

    DICE COEFFICIENT — SIMPLE EXPLANATION:
    "Of all the pixels that either the model or the truth marked as defects,
    how many did BOTH agree on?"
    - 1.0 = perfect agreement
    - 0.0 = no agreement at all
    """
    pred = (torch.sigmoid(predictions) > threshold).float()
    intersection = (pred * targets).sum()
    dice = (2.0 * intersection) / (pred.sum() + targets.sum() + 1e-8)
    return dice.item()


def compute_iou(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute IoU (Intersection over Union), also called Jaccard Index.

    IoU — SIMPLE EXPLANATION:
    Similar to Dice, but stricter. Imagine two circles overlapping:
    IoU = (area of overlap) / (total area covered by EITHER circle)
    It penalizes mistakes more harshly than Dice.
    """
    pred = (torch.sigmoid(predictions) > threshold).float()
    intersection = (pred * targets).sum()
    union = pred.sum() + targets.sum() - intersection
    iou = intersection / (union + 1e-8)
    return iou.item()


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler):
    """
    Train the model for one epoch (one pass through all training data).
    Returns average loss and Dice score for this epoch.
    """
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for images, masks in progress_bar:
        # Move data to GPU
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass with mixed precision (faster on modern GPUs)
        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            predictions = model(images)
            loss = criterion(predictions, masks)

        # Backward pass (compute gradients) and update weights
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Track metrics
        running_loss += loss.item()
        running_dice += compute_dice_score(predictions.detach(), masks)
        num_batches += 1

        progress_bar.set_postfix(
            loss=f"{running_loss/num_batches:.4f}",
            dice=f"{running_dice/num_batches:.4f}",
        )

    return running_loss / num_batches, running_dice / num_batches


def validate(model, val_loader, criterion, device):
    """
    Evaluate the model on the validation set (no gradient computation).
    Returns average loss, Dice score, and IoU.
    """
    model.eval()  # Set model to evaluation mode (disables dropout, uses running batch norm stats)
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    num_batches = 0

    with torch.no_grad():  # Don't compute gradients — saves memory and speeds up
        for images, masks in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                predictions = model(images)
                loss = criterion(predictions, masks)

            running_loss += loss.item()
            running_dice += compute_dice_score(predictions, masks)
            running_iou += compute_iou(predictions, masks)
            num_batches += 1

    avg_loss = running_loss / num_batches
    avg_dice = running_dice / num_batches
    avg_iou = running_iou / num_batches

    return avg_loss, avg_dice, avg_iou


def plot_training_curves(history: dict, save_path: str) -> None:
    """
    Plot training and validation loss/Dice curves.

    These curves help you see:
    - Is the model learning? (loss going down, Dice going up)
    - Is it overfitting? (training Dice keeps improving but validation stops)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice score curves
    axes[1].plot(epochs, history["train_dice"], "b-", label="Train Dice")
    axes[1].plot(epochs, history["val_dice"], "r-", label="Val Dice")
    axes[1].set_title("Dice Score over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # IoU curve (validation only)
    axes[2].plot(epochs, history["val_iou"], "g-", label="Val IoU")
    axes[2].set_title("IoU over Epochs")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nTraining curves saved to: {save_path}")


def train(
    num_epochs: int = 40,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    patience: int = 7,
    encoder_name: str = "resnet34",
):
    """
    Main training function — runs the complete training pipeline.

    Parameters
    ----------
    num_epochs : int
        Maximum number of epochs to train. We may stop earlier if
        early stopping triggers.
    batch_size : int
        Images per batch. Use 4 if you get out-of-memory errors.
    learning_rate : float
        How fast the model adjusts its weights. 1e-4 = 0.0001.
    patience : int
        Early stopping patience — stop if no improvement for this many epochs.
    encoder_name : str
        Backbone architecture. "resnet34" is our default.
    """
    # Setup
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "train.csv")
    image_dir = os.path.join(project_root, "data", "train_images")
    model_save_path = os.path.join(project_root, "models", "best_model.pth")
    curves_save_path = os.path.join(project_root, "models", "training_curves.png")
    history_save_path = os.path.join(project_root, "models", "training_history.json")

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Step 1: Prepare data
    print("\n--- Preparing Data ---")
    train_df, val_df, test_df = prepare_data_splits(csv_path)
    class_weights = get_class_weights(train_df).to(device)
    train_loader, val_loader, _ = create_dataloaders(
        train_df, val_df, test_df, image_dir,
        batch_size=batch_size, num_workers=2,
    )

    # Step 2: Create model
    print("\n--- Creating Model ---")
    model = create_model(encoder_name=encoder_name)
    model = model.to(device)

    # Step 3: Loss function, optimizer, scheduler
    criterion = CombinedLoss(class_weights=class_weights)

    # Adam optimizer — a good all-around optimizer that adjusts learning rate
    # automatically for each parameter
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Cosine annealing scheduler — gradually decreases learning rate
    # like cooling metal: fast changes at first, then finer adjustments
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # Mixed precision scaler — trains faster by using 16-bit numbers where possible (on CUDA)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Step 4: Training loop
    print(f"\n--- Training for up to {num_epochs} epochs ---")
    print(f"    Batch size: {batch_size}")
    print(f"    Learning rate: {learning_rate}")
    print(f"    Early stopping patience: {patience}")

    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "val_iou": [],
    }
    best_val_dice = 0.0
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # Validate
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Log metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        epoch_time = time.time() - epoch_start
        print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
                "encoder_name": encoder_name,
            }, model_save_path)
            print(f"  ★ New best model saved! (Dice: {val_dice:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n⚡ Early stopping triggered after {epoch} epochs!")
            print(f"   Best validation Dice: {best_val_dice:.4f}")
            break

    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best validation Dice score: {best_val_dice:.4f}")
    print(f"Model saved to: {model_save_path}")
    print(f"{'='*60}")

    # Save training history
    with open(history_save_path, "w") as f:
        json.dump(history, f)

    # Plot training curves
    plot_training_curves(history, curves_save_path)

    return model, history


if __name__ == "__main__":
    train()
