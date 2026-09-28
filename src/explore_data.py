"""
explore_data.py — Step 2: Load and explore the Severstal dataset
================================================================
This script:
  1. Loads the train.csv file (contains defect labels in RLE format)
  2. Decodes the RLE masks back into image masks
  3. Shows example images with colored defect overlays
  4. Prints how many images belong to each defect class

Run with:
    python src/explore_data.py

What you'll learn from running this:
  - What the steel images look like
  - What the 4 types of defects look like (colored overlays)
  - How imbalanced the classes are (some defects are much rarer than others)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path so we can import our utils module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import rle_to_mask, load_image, overlay_mask_on_image


def load_and_parse_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the train.csv and parse it into a useful format.

    The original CSV has columns:
      - ImageId_ClassId: e.g., "0002cc93b.jpg_1" (image name + defect class)
      - EncodedPixels: The RLE-encoded defect mask (or NaN if no defect)

    We split this into separate columns for easier use.
    """
    df = pd.read_csv(csv_path)

    # Split "ImageId_ClassId" into two separate columns
    df["ImageId"] = df["ImageId_ClassId"].apply(lambda x: x.split("_")[0])
    df["ClassId"] = df["ImageId_ClassId"].apply(lambda x: int(x.split("_")[1]))

    # Mark whether this row has a defect (EncodedPixels is not empty)
    df["HasDefect"] = df["EncodedPixels"].notna()

    return df


def print_class_distribution(df: pd.DataFrame) -> None:
    """
    Print how many images have each type of defect.
    This helps us understand CLASS IMBALANCE — when some defect types
    are much rarer than others, the model might struggle to learn them.
    """
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION — How many images per defect type?")
    print("=" * 60)

    total_images = df["ImageId"].nunique()
    print(f"\nTotal unique images: {total_images}")

    # Count images WITH a defect, grouped by class
    defect_df = df[df["HasDefect"]]
    class_counts = defect_df.groupby("ClassId")["ImageId"].nunique()

    # Count images with NO defects at all
    images_with_any_defect = defect_df["ImageId"].nunique()
    images_no_defect = total_images - images_with_any_defect

    print(f"Images with NO defects: {images_no_defect} ({100*images_no_defect/total_images:.1f}%)")
    print(f"Images with at least 1 defect: {images_with_any_defect} ({100*images_with_any_defect/total_images:.1f}%)")
    print()

    for class_id in sorted(class_counts.index):
        count = class_counts[class_id]
        pct = 100 * count / total_images
        bar = "█" * int(pct)
        print(f"  Class {class_id}: {count:>5} images ({pct:>5.1f}%) {bar}")

    print()
    print("NOTE: Class imbalance means some defect types are much rarer.")
    print("      We'll handle this later by weighting the loss function.")
    print("=" * 60)


def show_example_images(
    df: pd.DataFrame,
    image_dir: str,
    num_examples: int = 8,
    save_path: str = None,
) -> None:
    """
    Show a grid of example images with their defect masks overlaid.
    Each defect class gets a different color so you can tell them apart.
    """
    # Find images that have at least one defect (more interesting to look at)
    defect_images = df[df["HasDefect"]]["ImageId"].unique()

    if len(defect_images) == 0:
        print("No images with defects found!")
        return

    # Pick a few random examples
    np.random.seed(42)  # Fixed seed so we get the same examples every time
    selected = np.random.choice(
        defect_images, size=min(num_examples, len(defect_images)), replace=False
    )

    # Create a grid of subplots
    fig, axes = plt.subplots(num_examples, 2, figsize=(20, 3 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    # Color legend
    color_names = {1: "Red", 2: "Green", 3: "Blue", 4: "Yellow"}

    for idx, image_id in enumerate(selected):
        # Load the image
        img_path = os.path.join(image_dir, image_id)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        image = load_image(img_path)

        # Get all defect masks for this image
        image_rows = df[df["ImageId"] == image_id]
        masks = {}
        defect_classes = []
        for _, row in image_rows.iterrows():
            if row["HasDefect"]:
                mask = rle_to_mask(row["EncodedPixels"])
                masks[row["ClassId"]] = mask
                defect_classes.append(row["ClassId"])

        # Create the overlay
        overlay = overlay_mask_on_image(image, masks)

        # Show original image on the left
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f"Original: {image_id}", fontsize=10)
        axes[idx, 0].axis("off")

        # Show overlay on the right
        axes[idx, 1].imshow(overlay)
        class_str = ", ".join([f"Class {c} ({color_names[c]})" for c in defect_classes])
        axes[idx, 1].set_title(f"Defects: {class_str}", fontsize=10)
        axes[idx, 1].axis("off")

    # Add a legend
    legend_patches = [
        mpatches.Patch(color="red", label="Class 1"),
        mpatches.Patch(color="green", label="Class 2"),
        mpatches.Patch(color="blue", label="Class 3"),
        mpatches.Patch(color="yellow", label="Class 4"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=12)

    plt.suptitle("Steel Surface Defect Examples", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nVisualization saved to: {save_path}")

    plt.show()


def main():
    """Main entry point — run data exploration."""
    # Figure out paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "train.csv")
    image_dir = os.path.join(project_root, "data", "train_images")
    save_path = os.path.join(project_root, "data", "data_exploration.png")

    # Check that data exists
    if not os.path.exists(csv_path):
        print(f"ERROR: Could not find {csv_path}")
        print()
        print("Please download the Severstal dataset and place it in the data/ folder:")
        print("  1. Go to: https://www.kaggle.com/c/severstal-steel-defect-detection/data")
        print("  2. Download train.csv and train_images.zip")
        print("  3. Put train.csv in: data/train.csv")
        print("  4. Extract train_images.zip into: data/train_images/")
        print()
        print("Or use the Kaggle CLI:")
        print("  pip install kaggle")
        print("  kaggle competitions download -c severstal-steel-defect-detection -p data/")
        print("  # Then unzip the downloaded file into data/")
        sys.exit(1)

    # Step 1: Load and parse the CSV
    print("Loading train.csv...")
    df = load_and_parse_csv(csv_path)
    print(f"Loaded {len(df)} rows covering {df['ImageId'].nunique()} unique images")

    # Step 2: Print class distribution
    print_class_distribution(df)

    # Step 3: Show example images
    if os.path.exists(image_dir):
        print("\nGenerating example visualizations...")
        show_example_images(df, image_dir, num_examples=8, save_path=save_path)
    else:
        print(f"\nWARNING: Image directory not found: {image_dir}")
        print("Skipping visualization. Please extract train_images.zip first.")


if __name__ == "__main__":
    main()
