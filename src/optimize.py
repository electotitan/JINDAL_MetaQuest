"""
optimize.py — Step 7: Measure and improve inference speed
==========================================================
This script:
  1. Measures how fast the model processes one image (inference time)
  2. If too slow (>100ms), shows how to swap to a lighter backbone
  3. Explains quantization and pruning as future options

TARGET: < 100ms per image for real-time factory inspection

Run with:
    python src/optimize.py
"""

import os
import sys
import time
import torch
import numpy as np
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import create_model
from src.dataset import IMG_HEIGHT, IMG_WIDTH


def benchmark_inference(
    model,
    device: torch.device,
    input_size: tuple = (1, 3, IMG_HEIGHT, IMG_WIDTH),
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    """
    Measure how fast the model processes one image.

    We run it many times and report the average, because individual
    runs can vary due to GPU scheduling, caching, etc.

    Parameters
    ----------
    num_warmup : int
        Run the model this many times first without measuring.
        This "warms up" the GPU (loads kernels into cache).
    num_runs : int
        Number of timed runs to average over.
    """
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)

    # Warm up (first few runs are always slower)
    print(f"Warming up ({num_warmup} runs)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()  # Wait for GPU to finish

    # Timed runs
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            with autocast(device_type=device.type):
                _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds

    results = {
        "device": str(device),
        "mean_ms": np.mean(times),
        "median_ms": np.median(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "std_ms": np.std(times),
        "fps": 1000 / np.mean(times),  # Frames per second
    }
    return results


def print_benchmark_results(results: dict, model_name: str) -> None:
    """Print benchmark results in a nice format."""
    print(f"\n  {model_name} on {results['device']}:")
    print(f"    Mean:   {results['mean_ms']:.1f} ms/image")
    print(f"    Median: {results['median_ms']:.1f} ms/image")
    print(f"    Min:    {results['min_ms']:.1f} ms | Max: {results['max_ms']:.1f} ms")
    print(f"    FPS:    {results['fps']:.1f} frames/second")

    if results["mean_ms"] < 100:
        print(f"    ✓ PASSES real-time target (< 100 ms/image)")
    else:
        print(f"    ✗ Too slow for real-time (target: < 100 ms/image)")


def compare_backbones(device: torch.device) -> None:
    """
    Compare inference speed of different encoder backbones.

    If ResNet34 is too slow, we can swap to a lighter backbone:
    - MobileNetV2: Designed for mobile phones — very fast, slightly less accurate
    - EfficientNet-B0: Good speed/accuracy tradeoff
    """
    backbones = {
        "ResNet34 (default)": "resnet34",
        "MobileNetV2 (fast)": "mobilenet_v2",
        "EfficientNet-B0 (balanced)": "efficientnet-b0",
    }

    print("\n" + "=" * 60)
    print("BACKBONE COMPARISON — Speed vs Model Size")
    print("=" * 60)

    all_results = {}
    for display_name, encoder_name in backbones.items():
        try:
            model = create_model(encoder_name=encoder_name)
            model = model.to(device)
            results = benchmark_inference(model, device, num_runs=50)
            print_benchmark_results(results, display_name)
            all_results[display_name] = results
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None
        except Exception as e:
            print(f"\n  {display_name}: FAILED — {e}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Backbone':<30} {'Speed (ms)':<12} {'FPS':<10} {'Passes?':<10}")
    print(f"{'-'*60}")
    for name, res in all_results.items():
        passes = "✓ Yes" if res["mean_ms"] < 100 else "✗ No"
        print(f"{name:<30} {res['mean_ms']:<12.1f} {res['fps']:<10.1f} {passes:<10}")
    print(f"{'='*60}")


def explain_optimization_techniques():
    """
    Explain advanced optimization techniques in simple terms.
    These are NOT implemented here — just explained for future reference.
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║  FUTURE SPEED-UP OPTIONS (explained, not implemented)       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. QUANTIZATION                                             ║
║  ──────────────                                              ║
║  What: Using simpler numbers inside the model.               ║
║                                                              ║
║  Analogy: Imagine measuring a table's length.                ║
║  Full precision: 120.3456789 cm (very precise but slow)      ║
║  Quantized:      120.3 cm (good enough, much faster!)        ║
║                                                              ║
║  The model normally uses 32-bit floating point numbers.      ║
║  Quantization converts them to 8-bit integers (4x smaller).  ║
║  This makes the model ~2-4x faster with only 1-2% accuracy  ║
║  drop.                                                       ║
║                                                              ║
║  How: torch.quantization.quantize_dynamic(model)             ║
║                                                              ║
║  ────────────────────────────────────────────────────────     ║
║                                                              ║
║  2. PRUNING                                                  ║
║  ────────                                                    ║
║  What: Removing unimportant connections in the neural net.   ║
║                                                              ║
║  Analogy: A tree has many branches. Some branches are dead   ║
║  or barely contribute. Pruning cuts those off, leaving a     ║
║  lighter, faster tree that still produces good fruit.        ║
║                                                              ║
║  In a neural network, many weights (connections) are close   ║
║  to zero and don't contribute much. Removing them makes      ║
║  the model smaller and faster.                               ║
║                                                              ║
║  Typical speedup: 2-3x with ~1-3% accuracy drop.            ║
║                                                              ║
║  How: torch.nn.utils.prune.l1_unstructured(layer, 'weight') ║
║                                                              ║
║  ────────────────────────────────────────────────────────     ║
║                                                              ║
║  3. ONNX EXPORT + TensorRT (most advanced)                   ║
║  ──────────────────────────                                  ║
║  What: Converting the model to a format optimized for your   ║
║  specific GPU. NVIDIA's TensorRT can fuse operations and     ║
║  optimize memory access patterns.                            ║
║                                                              ║
║  Typical speedup: 3-5x on NVIDIA GPUs.                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    """Run inference benchmarks and comparisons."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Benchmark current model (ResNet34)
    print("\n--- Benchmarking Default Model (ResNet34) ---")
    model = create_model(encoder_name="resnet34")
    model = model.to(device)
    results = benchmark_inference(model, device)
    print_benchmark_results(results, "ResNet34 U-Net")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Compare backbones
    compare_backbones(device)

    # Explain future optimization options
    explain_optimization_techniques()

    print("\n✓ Speed optimization analysis complete!")


if __name__ == "__main__":
    main()
