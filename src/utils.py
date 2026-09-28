"""
utils.py — Helper functions used across the project.
=====================================================
The most important function here is `rle_to_mask`, which converts
the compressed defect-mask format (RLE) back into a normal image mask.

KEY TERM — RLE (Run-Length Encoding):
    Instead of storing every single pixel value (0 or 1), the dataset
    stores pairs of numbers: "start at pixel X, then fill the next Y pixels."
    Example: "100 5" means "starting at pixel 100, color 5 pixels."
    This saves a LOT of space. We need to "decode" this back into a
    full pixel grid (a 2D array) so we can use it for training.
"""

import numpy as np
import cv2
import os


def rle_to_mask(rle_string: str, height: int = 256, width: int = 1600) -> np.ndarray:
    """
    Convert a Run-Length Encoded (RLE) string into a binary mask.

    Parameters
    ----------
    rle_string : str
        The RLE-encoded string from the CSV file.
        Format: "start1 length1 start2 length2 ..."
        Each pair means: starting at pixel `start`, fill `length` pixels with 1.
    height : int
        Image height (256 for the Severstal dataset).
    width : int
        Image width (1600 for the Severstal dataset).

    Returns
    -------
    mask : numpy array of shape (height, width)
        Binary mask where 1 = defect pixel, 0 = normal pixel.

    Example
    -------
    >>> mask = rle_to_mask("1 3 10 5")
    >>> # Pixels 1,2,3 and 10,11,12,13,14 are set to 1
    """
    # Start with an all-zero (no-defect) mask
    mask = np.zeros(height * width, dtype=np.uint8)

    # If there's no RLE data (no defect), return the empty mask
    if not isinstance(rle_string, str) or rle_string.strip() == "":
        return mask.reshape((height, width), order="F")

    # Parse the RLE string into pairs of (start, length)
    values = rle_string.split()
    starts = values[0::2]  # Every other value starting from index 0
    lengths = values[1::2]  # Every other value starting from index 1

    for start, length in zip(starts, lengths):
        start = int(start) - 1  # RLE is 1-indexed, Python is 0-indexed
        length = int(length)
        mask[start : start + length] = 1

    # The mask is stored column-first (Fortran order), so we reshape accordingly
    # "order='F'" means the pixels go down each column, then move to the next column
    return mask.reshape((height, width), order="F")


def mask_to_rle(mask: np.ndarray) -> str:
    """
    Convert a binary mask back to RLE string.
    This is the reverse of rle_to_mask — useful for submitting predictions.

    Parameters
    ----------
    mask : numpy array of shape (height, width)
        Binary mask where 1 = defect, 0 = normal.

    Returns
    -------
    rle_string : str
        RLE-encoded string.
    """
    # Flatten the mask in column-first order (same as the dataset uses)
    pixels = mask.flatten(order="F")

    # Pad with zeros at start and end to detect transitions
    pixels = np.concatenate([[0], pixels, [0]])

    # Find where pixels change from 0→1 (starts) and 1→0 (ends)
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs = runs.reshape(-1, 2)  # Pair up starts and ends

    # Convert to RLE format: start (1-indexed) and length
    runs[:, 0] += 1  # Convert to 1-indexed
    runs[:, 1] -= runs[:, 0]  # Convert end position to length
    runs[:, 1] += 1

    return " ".join(runs.flatten().astype(str))


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk.

    Returns the image as a numpy array in RGB format (not BGR, which is
    OpenCV's default). Most deep learning models expect RGB.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    # OpenCV loads images as BGR (Blue, Green, Red).
    # We convert to RGB (Red, Green, Blue) because that's what PyTorch models expect.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def overlay_mask_on_image(
    image: np.ndarray,
    masks: dict,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Draw colored defect masks on top of the original image.

    Parameters
    ----------
    image : numpy array (H, W, 3)
        The original RGB image.
    masks : dict
        Keys are class IDs (1-4), values are binary masks (H, W).
    alpha : float
        Transparency of the overlay (0 = invisible, 1 = fully opaque).

    Returns
    -------
    overlay : numpy array (H, W, 3)
        The image with colored masks drawn on top.
    """
    # One distinct color per defect class (RGB format)
    colors = {
        1: (255, 0, 0),      # Class 1 → Red
        2: (0, 255, 0),      # Class 2 → Green
        3: (0, 0, 255),      # Class 3 → Blue
        4: (255, 255, 0),    # Class 4 → Yellow
    }

    overlay = image.copy()
    for class_id, mask in masks.items():
        if mask.sum() == 0:
            continue  # Skip if no defect pixels for this class
        color = colors.get(class_id, (255, 255, 255))
        # Where the mask is 1, blend the color onto the image
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask == 1,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c],
            )
    return overlay.astype(np.uint8)


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
