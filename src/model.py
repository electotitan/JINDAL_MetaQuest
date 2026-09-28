"""
model.py — Step 4: U-Net Segmentation Model
=============================================
This file creates our defect detection model using a well-tested library
called `segmentation-models-pytorch` (SMP).

WHAT IS SEGMENTATION?
  Regular image classification says "this image contains a cat."
  Segmentation says "THESE SPECIFIC PIXELS are the cat."
  For us: it says "THESE SPECIFIC PIXELS are defective."

THE MODEL WE USE: U-Net with ResNet34 encoder

  Think of it like this:

  ENCODER (the "shrinking" half):
  ┌─────────────────────────────────────────┐
  │  Input Image (256 x 512 x 3)           │
  │    ↓                                    │
  │  Layer 1: Finds edges                  │   ← "I see lines and borders"
  │    ↓ (shrink)                           │
  │  Layer 2: Finds textures               │   ← "I see rough vs smooth areas"
  │    ↓ (shrink)                           │
  │  Layer 3: Finds patterns               │   ← "I see scratches, spots"
  │    ↓ (shrink)                           │
  │  Layer 4: Understands defect types     │   ← "I think this is a crack"
  └─────────────────────────────────────────┘

  DECODER (the "expanding" half):
  ┌─────────────────────────────────────────┐
  │  Takes the compressed understanding     │
  │    ↓ (expand)                           │
  │  Maps it back to full image size       │
  │    ↓ (expand)                           │
  │  Says exactly WHICH pixels are defects │
  │    ↓                                    │
  │  Output: 4 masks (256 x 512 x 4)      │
  └─────────────────────────────────────────┘

  SKIP CONNECTIONS (the "U" in U-Net):
  The encoder sends information directly to the decoder at each level.
  This is like leaving bookmarks — the decoder can look back at the
  fine details (edges, textures) it otherwise would have lost during
  the shrinking process. This is why U-Net produces very precise masks.

  PRETRAINED ON IMAGENET:
  The encoder (ResNet34) has already seen millions of everyday photos
  and learned to recognize edges, textures, and patterns. Instead of
  learning from scratch, we reuse this knowledge and just teach it
  "which patterns mean steel defects." This is called TRANSFER LEARNING
  and it makes training MUCH faster and better.
"""

import segmentation_models_pytorch as smp


def create_model(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    num_classes: int = 4,
    activation: str = None,
) -> smp.Unet:
    """
    Create a U-Net segmentation model.

    Parameters
    ----------
    encoder_name : str
        Which pretrained network to use as the encoder.
        - "resnet34": Good balance of speed and accuracy (our default)
        - "mobilenet_v2": Faster but slightly less accurate
        - "efficientnet-b0": Another lightweight option
    encoder_weights : str
        "imagenet" = use weights pretrained on ImageNet (recommended)
        None = start from scratch (not recommended — much slower to train)
    num_classes : int
        Number of defect classes (4 for our dataset)
    activation : str
        Final activation function. We use None here and apply sigmoid
        in the loss function instead (more numerically stable).

    Returns
    -------
    model : smp.Unet
        The segmentation model ready for training
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,          # 3 color channels (RGB)
        classes=num_classes,    # 4 output masks (one per defect type)
        activation=activation,  # Applied in loss function instead
    )

    # Print a summary of the model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: U-Net with {encoder_name} encoder")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Output classes:       {num_classes}")

    return model


# ============================================================
# Quick test — verify the model can process a sample input
# ============================================================
if __name__ == "__main__":
    import torch

    print("Creating U-Net model with ResNet34 encoder...\n")
    model = create_model()

    # Create a fake batch of 2 images (2, 3, 256, 512)
    dummy_input = torch.randn(2, 3, 256, 512)
    print(f"\nTest input shape:  {dummy_input.shape}")

    # Run it through the model
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Test output shape: {output.shape}")
    # Expected: (2, 4, 256, 512) — 2 images, 4 classes, full resolution

    print(f"\nOutput value range: [{output.min():.3f}, {output.max():.3f}]")
    print("(These are raw logits — we apply sigmoid later to get probabilities)")
    print("\n✓ Model created and tested successfully!")
