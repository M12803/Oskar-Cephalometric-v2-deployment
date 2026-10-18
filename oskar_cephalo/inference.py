"""Inference Engine wrapping the ONNX Runtime session."""

import os
from pathlib import Path
from typing import Union, Dict, Any, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

from oskar_cephalo.preprocessor import CephaloPreprocessor
from oskar_cephalo.postprocessor import CephaloPostprocessor


class OskarCephaloDetector:
    """Production Inference Engine for Oskar Cephalo ONNX model."""

    DEFAULT_MODEL_FILENAME = "oskar_cephalo.onnx"

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize detector with ONNX model path.
        If model_path is None, automatically searches standard project locations.
        """
        self.model_path = self._resolve_model_path(model_path)
        print(f"[OskarCephaloDetector] Loading model from: {self.model_path}")

        # Initialize ONNX Runtime Session with optimal settings
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=options,
            providers=['CPUExecutionProvider']
        )
        print("[OskarCephaloDetector] Model loaded successfully and ready for inference!")

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        if model_path and os.path.exists(model_path):
            return str(Path(model_path).resolve())

        search_paths = [
            Path(self.DEFAULT_MODEL_FILENAME),
            Path(__file__).parent.parent / self.DEFAULT_MODEL_FILENAME,
            Path("ockar cephalo.onnx"),
            Path(__file__).parent.parent / "ockar cephalo.onnx",
        ]
        for p in search_paths:
            if p.exists():
                return str(p.resolve())

        raise FileNotFoundError(
            f"Could not find {self.DEFAULT_MODEL_FILENAME}. Please ensure the ONNX model is in the root directory."
        )

    def predict(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        point_size: int = 6,
        show_labels: bool = True,
        draw_lines: bool = True
    ) -> Dict[str, Any]:
        """
        Run end-to-end inference on a lateral cephalogram.

        Returns:
            {
                "landmarks": List of dicts (symbol, name, x, y, confidence, description),
                "cvm_stage": Dict (stage, code, phase, description, confidence),
                "image_width": orig_w,
                "image_height": orig_h,
                "rendered_image": numpy array RGB with landmarks and lines
            }
        """
        # 1. Preprocessing
        input_tensor, orig_w, orig_h, original_rgb = CephaloPreprocessor.process(image_input)

        # 2. ONNX Execution
        outputs = self.session.run(None, {'input_image': input_tensor})
        heatmaps, cvm_logits = outputs[0], outputs[1]

        # 3. Postprocessing
        landmarks = CephaloPostprocessor.extract_coordinates(heatmaps, orig_w, orig_h)
        cvm_info = CephaloPostprocessor.parse_cvm(cvm_logits)

        # 4. Visualization
        rendered_image = CephaloPostprocessor.render_overlay(
            original_rgb,
            landmarks,
            point_size=point_size,
            show_labels=show_labels,
            draw_lines=draw_lines
        )

        return {
            "landmarks": landmarks,
            "cvm_stage": cvm_info,
            "image_width": orig_w,
            "image_height": orig_h,
            "rendered_image": rendered_image
        }
