"""Image preprocessing pipeline matching model training."""

from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image


class CephaloPreprocessor:
    """Preprocesses lateral cephalometric radiographs for the HRNet ONNX model."""

    INPUT_WIDTH = 768
    INPUT_HEIGHT = 864

    @classmethod
    def process(
        cls,
        image_input: Union[str, np.ndarray, Image.Image],
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> Tuple[np.ndarray, int, int, np.ndarray]:
        """
        Preprocess input image:
        1. Convert to Grayscale
        2. Apply CLAHE contrast enhancement
        3. Aspect-ratio preserving resize to (768, 864)
        4. Normalize to [0.0, 1.0]

        Returns:
            input_tensor: (1, 1, 864, 768) float32 numpy array
            orig_w: Original image width
            orig_h: Original image height
            original_rgb: (orig_h, orig_w, 3) uint8 numpy array for rendering
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not read image from path: {image_input}")
            original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif isinstance(image_input, Image.Image):
            original_rgb = np.array(image_input.convert("RGB"))
            gray = np.array(image_input.convert("L"))
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                original_rgb = image_input.copy()
                gray = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_input.copy()
                original_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        orig_h, orig_w = gray.shape[:2]

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(gray)

        # Resize to fixed input resolution
        resized = cv2.resize(enhanced, (cls.INPUT_WIDTH, cls.INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0

        # Shape: (1, 1, 864, 768)
        input_tensor = normalized[np.newaxis, np.newaxis, :, :]

        return input_tensor, orig_w, orig_h, original_rgb
