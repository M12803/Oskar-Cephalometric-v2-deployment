'''Test script for Oskar Cephalo ONNX inference'''
import os
from oskar_cephalo.inference import OskarCephaloDetector


def main():
    # Locate a sample image – you can replace this path with any cephalogram image.
    sample_image = os.path.join(os.path.dirname(__file__), "samples", "sample1.png")
    if not os.path.exists(sample_image):
        print(f"Sample image not found at {sample_image}. Please add an example image to the 'samples' folder.")
        return

    detector = OskarCephaloDetector()
    result = detector.predict(sample_image)

    print("--- Inference Result ---")
    print(f"CVM Stage: {result['cvm_stage'].get('stage', 'N/A')} (code: {result['cvm_stage'].get('code', 'N/A')})")
    print("Landmarks (x, y):")
    for lm in result["landmarks"]:
        print(f"  {lm['symbol']}: ({lm['x']:.1f}, {lm['y']:.1f}) confidence={lm.get('confidence', 'N/A'):.2f}")
    # Optionally, save the rendered overlay image
    overlay_path = os.path.join(os.path.dirname(__file__), "samples", "overlay.png")
    from PIL import Image
    Image.fromarray(result["rendered_image"]).save(overlay_path)
    print(f"Overlay image saved to {overlay_path}")


if __name__ == "__main__":
    main()
