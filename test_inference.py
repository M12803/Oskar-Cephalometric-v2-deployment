#!/usr/bin/env python3
"""
Oskar Cephalo v2 — Quick Inference Test Script.

Usage:
    python test_inference.py
    python test_inference.py --image path/to/xray.png
    python test_inference.py --save-overlay

This script verifies that the ONNX model loads correctly and produces
valid predictions. Use it to confirm your installation is working.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from oskar_cephalo import OskarCephaloDetector


def find_sample_image() -> str:
    """Find a sample image in the samples/ directory."""
    samples_dir = Path(__file__).parent / "samples"
    if samples_dir.exists():
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            files = list(samples_dir.glob(ext))
            if files:
                return str(files[0])
    return ""


def main():
    parser = argparse.ArgumentParser(description="Oskar Cephalo v2 — Test Inference")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a cephalometric X-ray image")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the ONNX model file")
    parser.add_argument("--save-overlay", action="store_true",
                        help="Save the rendered overlay image")
    parser.add_argument("--output", type=str, default="test_overlay.png",
                        help="Output path for the overlay image")
    args = parser.parse_args()

    # Find image
    image_path = args.image or find_sample_image()
    if not image_path or not os.path.exists(image_path):
        print("❌ No image found. Please provide one with --image <path>")
        print("   Or place a sample image in the 'samples/' folder.")
        sys.exit(1)

    print("=" * 60)
    print("  🦷 Oskar Cephalo v2 — Inference Test")
    print("=" * 60)
    print(f"  Image: {image_path}")
    print()

    # Load model
    t0 = time.time()
    detector = OskarCephaloDetector(model_path=args.model)
    load_time = time.time() - t0
    print(f"  ⏱️  Model loaded in {load_time:.2f}s")
    print()

    # Run inference
    t0 = time.time()
    result = detector.predict(image_path)
    infer_time = time.time() - t0
    print(f"  ⏱️  Inference completed in {infer_time:.2f}s")
    print()

    # Display CVM stage
    cvm = result["cvm_stage"]
    print(f"  🦴 CVM Stage: {cvm['code']} ({cvm['phase']}) — Confidence: {cvm['confidence']}%")
    print(f"     {cvm['description']}")
    print()

    # Display landmarks
    print(f"  📍 Detected {len(result['landmarks'])} landmarks:")
    print(f"  {'#':<4} {'Symbol':<8} {'Name':<25} {'X':>8} {'Y':>8} {'Conf':>8}")
    print(f"  {'─'*4} {'─'*8} {'─'*25} {'─'*8} {'─'*8} {'─'*8}")
    for lm in result["landmarks"]:
        print(f"  {lm['index']:<4} {lm['symbol']:<8} {lm['name']:<25} "
              f"{lm['x']:>8.1f} {lm['y']:>8.1f} {lm['confidence']:>8.4f}")

    # Save overlay
    if args.save_overlay:
        from PIL import Image
        overlay = Image.fromarray(result["rendered_image"])
        overlay.save(args.output)
        print(f"\n  💾 Overlay saved to: {args.output}")

    print()
    print("  ✅ All checks passed! Model is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
