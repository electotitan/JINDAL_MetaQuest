"""
evaluate.py — Step 6: Evaluate the trained model
==================================================
This script:
  1. Loads the best trained model
  2. Runs it on the held-out test set
  3. Computes metrics: Dice, IoU, Recall, Precision, False-alarm rate
  4. Shows side-by-side visualizations (image | true mask | predicted mask)
  5. Optionally runs 5-fold cross-validation

KEY TERMS:
  - RECALL (Sensitivity): "Of all REAL defects, how many did we CATCH?"
    High recall = we rarely miss a defect. We want ≥ 90%.
  - PRECISION: "Of everything we FLAGGED as a defect, how much was REAL?"
    High precision = few false alarms.
  - FALSE-ALARM RATE: "Of all normal areas, how many did we wrongly flag?"
    This is (1 - Specificity). We want this LOW.
  - CONFUSION: High recall + low precision = "catches everything but also
    raises lots of false alarms." We want BOTH high.

Run with:
    python src/evaluate.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from sklearn.model_selection import KFold
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import (
    prepare_data_splits, create_dataloaders,
    get_class_weights, SteelDataset, get_augmentation_pipeline,
    IMG_HEIGHT, IMG_WIDTH, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES,
)
from src.model import create_model
from src.train import CombinedLoss


def load_best_model(model_path: str, device: torch.device):
    """Load the best saved model checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    encoder_name = checkpoint.get("encoder_name", "resnet34")
    model = create_model(encoder_name=encoder_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Val Dice at save time: {checkpoint['val_dice']:.4f}")
    return model


def compute_per_class_metrics(
    model, data_loader, device, threshold: float = 0.5
):
    """
    Compute detailed metrics for each defect class.

    Returns a dict with per-class and overall metrics.
    """
    # Accumulators for each class
    class_tp = np.zeros(NUM_CLASSES)   # True Positives (correctly found defects)
    class_fp = np.zeros(NUM_CLASSES)   # False Positives (wrongly flagged as defect)
    class_fn = np.zeros(NUM_CLASSES)   # False Negatives (missed real defects)
    class_tn = np.zeros(NUM_CLASSES)   # True Negatives (correctly said "no defect")
    class_dice_sum = np.zeros(NUM_CLASSES)
    class_iou_sum = np.zeros(NUM_CLASSES)
    num_samples = 0

    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            with autocast(device_type="cuda"):
                predictions = model(images)

            # Convert to binary predictions
            pred_binary = (torch.sigmoid(predictions) > threshold).float()

            for c in range(NUM_CLASSES):
                pred_c = pred_binary[:, c].cpu().numpy().flatten()
                true_c = masks[:, c].cpu().numpy().flatten()

                # Count TP, FP, FN, TN
                tp = ((pred_c == 1) & (true_c == 1)).sum()
                fp = ((pred_c == 1) & (true_c == 0)).sum()
                fn = ((pred_c == 0) & (true_c == 1)).sum()
                tn = ((pred_c == 0) & (true_c == 0)).sum()

                class_tp[c] += tp
                class_fp[c] += fp
                class_fn[c] += fn
                class_tn[c] += tn

                # Dice for this batch
                intersection = tp
                dice = (2 * intersection) / (pred_c.sum() + true_c.sum() + 1e-8)
                class_dice_sum[c] += dice

                # IoU for this batch
                union = tp + fp + fn
                iou = intersection / (union + 1e-8)
                class_iou_sum[c] += iou

            num_samples += 1

    # Compute final metrics
    results = {}
    for c in range(NUM_CLASSES):
        tp, fp, fn, tn = class_tp[c], class_fp[c], class_fn[c], class_tn[c]

        recall = tp / (tp + fn + 1e-8)          # How many real defects we caught
        precision = tp / (tp + fp + 1e-8)       # How many of our flags were real
        false_alarm_rate = fp / (fp + tn + 1e-8) # Wrong flags / all normal areas
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results[f"class_{c+1}"] = {
            "dice": class_dice_sum[c] / num_samples,
            "iou": class_iou_sum[c] / num_samples,
            "recall": recall,
            "precision": precision,
            "false_alarm_rate": false_alarm_rate,
            "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }

    # Overall averages
    results["overall"] = {
        "dice": np.mean([results[f"class_{c+1}"]["dice"] for c in range(NUM_CLASSES)]),
        "iou": np.mean([results[f"class_{c+1}"]["iou"] for c in range(NUM_CLASSES)]),
        "recall": np.mean([results[f"class_{c+1}"]["recall"] for c in range(NUM_CLASSES)]),
        "precision": np.mean([results[f"class_{c+1}"]["precision"] for c in range(NUM_CLASSES)]),
        "false_alarm_rate": np.mean([results[f"class_{c+1}"]["false_alarm_rate"] for c in range(NUM_CLASSES)]),
    }

    return results


def print_metrics_table(results: dict) -> None:
    """Print metrics in a nice table format."""
    print("\n" + "=" * 85)
    print("EVALUATION RESULTS")
    print("=" * 85)
    print(f"{'Metric':<20} {'Class 1':>10} {'Class 2':>10} {'Class 3':>10} {'Class 4':>10} {'Overall':>10}")
    print("-" * 85)

    for metric in ["dice", "iou", "recall", "precision", "false_alarm_rate"]:
        display_name = metric.replace("_", " ").title()
        values = [results[f"class_{c+1}"][metric] for c in range(NUM_CLASSES)]
        overall = results["overall"][metric]
        print(f"{display_name:<20} {values[0]:>10.4f} {values[1]:>10.4f} "
              f"{values[2]:>10.4f} {values[3]:>10.4f} {overall:>10.4f}")

    print("=" * 85)

    # Check targets
    overall_recall = results["overall"]["recall"]
    if overall_recall >= 0.9:
        print(f"✓ Recall target MET: {overall_recall:.4f} ≥ 0.90")
    else:
        print(f"✗ Recall target NOT met: {overall_recall:.4f} < 0.90")
        print("  Consider: more training epochs, lower threshold, or more augmentation")


def visualize_predictions(
    model, data_loader, device, num_examples: int = 6, save_path: str = None
):
    """
    Show side-by-side comparisons: Original | True Mask | Predicted Mask.
    This lets you visually check if the model's predictions look reasonable.
    """
    model.eval()
    images_shown = 0

    fig, axes = plt.subplots(num_examples, 3, figsize=(18, 4 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    colors = [(1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5), (1, 1, 0, 0.5)]
    class_names = ["Class 1", "Class 2", "Class 3", "Class 4"]

    with torch.no_grad():
        for images, masks in data_loader:
            images_gpu = images.to(device)
            with autocast(device_type="cuda"):
                predictions = model(images_gpu)
            pred_binary = (torch.sigmoid(predictions) > 0.5).cpu().numpy()
            masks_np = masks.numpy()
            images_np = images.numpy()

            for i in range(images_np.shape[0]):
                if images_shown >= num_examples:
                    break

                # Denormalize image for display
                img = images_np[i].transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
                img = np.clip(img, 0, 1)

                # Original image
                axes[images_shown, 0].imshow(img)
                axes[images_shown, 0].set_title("Original Image")
                axes[images_shown, 0].axis("off")

                # True masks (all classes overlaid)
                true_overlay = img.copy()
                for c in range(NUM_CLASSES):
                    mask = masks_np[i, c]
                    if mask.sum() > 0:
                        for ch in range(3):
                            true_overlay[:, :, ch] = np.where(
                                mask > 0.5,
                                true_overlay[:, :, ch] * 0.6 + colors[c][ch] * 0.4,
                                true_overlay[:, :, ch],
                            )
                axes[images_shown, 1].imshow(np.clip(true_overlay, 0, 1))
                axes[images_shown, 1].set_title("Ground Truth Masks")
                axes[images_shown, 1].axis("off")

                # Predicted masks
                pred_overlay = img.copy()
                for c in range(NUM_CLASSES):
                    mask = pred_binary[i, c]
                    if mask.sum() > 0:
                        for ch in range(3):
                            pred_overlay[:, :, ch] = np.where(
                                mask > 0.5,
                                pred_overlay[:, :, ch] * 0.6 + colors[c][ch] * 0.4,
                                pred_overlay[:, :, ch],
                            )
                axes[images_shown, 2].imshow(np.clip(pred_overlay, 0, 1))
                axes[images_shown, 2].set_title("Predicted Masks")
                axes[images_shown, 2].axis("off")

                images_shown += 1

            if images_shown >= num_examples:
                break

    # Legend
    legend_patches = [
        mpatches.Patch(color=colors[i][:3], label=class_names[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=12)

    plt.suptitle("Model Predictions vs Ground Truth", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Prediction visualization saved to: {save_path}")
    plt.close()


def cross_validate(
    csv_path: str, image_dir: str, n_folds: int = 5,
    num_epochs: int = 15, batch_size: int = 8,
):
    """
    5-fold cross-validation to check if results are stable.

    WHY CROSS-VALIDATION?
    One train/test split might be "lucky" (easy test set) or "unlucky"
    (hard test set). Cross-validation trains 5 separate models on 5
    different splits and averages the results. If the scores are similar
    across folds, we can trust the results. If they vary wildly, something
    is wrong.

    NOTE: This trains 5 models, so it takes ~5x longer. We use fewer
    epochs (15) to keep it manageable.
    """
    print(f"\n{'='*60}")
    print(f"5-FOLD CROSS-VALIDATION (reduced epochs = {num_epochs})")
    print(f"{'='*60}")

    # Load full dataset
    df = pd.read_csv(csv_path)
    df["ImageId"] = df["ImageId_ClassId"].apply(lambda x: x.split("_")[0])
    df["ClassId"] = df["ImageId_ClassId"].apply(lambda x: int(x.split("_")[1]))
    df["HasDefect"] = df["EncodedPixels"].notna()

    unique_images = df["ImageId"].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_images)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        train_images = unique_images[train_idx]
        val_images = unique_images[val_idx]

        train_df = df[df["ImageId"].isin(train_images)].reset_index(drop=True)
        val_df = df[df["ImageId"].isin(val_images)].reset_index(drop=True)

        # Create data loaders
        from torch.utils.data import DataLoader
        train_dataset = SteelDataset(train_df, image_dir, phase="train")
        val_dataset = SteelDataset(val_df, image_dir, phase="val")
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)

        # Create model
        model = create_model(encoder_name="resnet34")
        model = model.to(device)

        class_weights = get_class_weights(train_df).to(device)
        criterion = CombinedLoss(class_weights=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler()

        # Quick training
        from src.train import train_one_epoch, validate
        best_dice = 0
        for epoch in range(1, num_epochs + 1):
            train_loss, train_dice = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)
            if val_dice > best_dice:
                best_dice = val_dice
            print(f"  Epoch {epoch}: Train Dice={train_dice:.4f}, Val Dice={val_dice:.4f}")

        fold_results.append(best_dice)
        print(f"  Fold {fold + 1} best Dice: {best_dice:.4f}")

        # Clean up GPU memory
        del model, optimizer, criterion
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for i, dice in enumerate(fold_results):
        print(f"  Fold {i+1}: Dice = {dice:.4f}")
    print(f"\n  Mean Dice: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"{'='*60}")

    if np.std(fold_results) < 0.05:
        print("✓ Results are stable across folds (low variance)")
    else:
        print("⚠ Results vary across folds — consider more data or different augmentation")

    return fold_results


def main():
    """Main evaluation pipeline."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "train.csv")
    image_dir = os.path.join(project_root, "data", "train_images")
    model_path = os.path.join(project_root, "models", "best_model.pth")
    vis_save_path = os.path.join(project_root, "models", "predictions_vs_truth.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check model exists
    if not os.path.exists(model_path):
        print(f"ERROR: No trained model found at {model_path}")
        print("Run 'python src/train.py' first to train the model.")
        sys.exit(1)

    # Load model
    print("\n--- Loading Best Model ---")
    model = load_best_model(model_path, device)

    # Prepare test data
    print("\n--- Preparing Test Data ---")
    _, _, test_df = prepare_data_splits(csv_path)
    from torch.utils.data import DataLoader
    test_dataset = SteelDataset(test_df, image_dir, phase="test")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Compute metrics
    print("\n--- Computing Metrics on Test Set ---")
    results = compute_per_class_metrics(model, test_loader, device)
    print_metrics_table(results)

    # Visualize predictions
    print("\n--- Generating Prediction Visualizations ---")
    visualize_predictions(model, test_loader, device, num_examples=6, save_path=vis_save_path)

    # Cross-validation (optional — takes a while)
    print("\n--- Running 5-Fold Cross-Validation ---")
    print("(This trains 5 models with reduced epochs. Will take ~15-20 minutes)")
    response = input("Run cross-validation? (y/n): ").strip().lower()
    if response == "y":
        cross_validate(csv_path, image_dir, n_folds=5, num_epochs=15, batch_size=8)
    else:
        print("Skipping cross-validation.")

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
