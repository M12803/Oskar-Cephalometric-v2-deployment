"""Oskar Cephalo v2 - Interactive Web Application.

Run locally:
    python app.py
"""

from pathlib import Path
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from oskar_cephalo import OskarCephaloDetector

# Initialize inference detector
detector = OskarCephaloDetector()


def predict(
    image: Image.Image,
    point_size: int = 6,
    show_labels: bool = True,
    draw_lines: bool = True
):
    if image is None:
        return None, "Please upload an X-ray image.", None

    results = detector.predict(
        image_input=image,
        point_size=point_size,
        show_labels=show_labels,
        draw_lines=draw_lines
    )

    cvm = results["cvm_stage"]
    cvm_markdown = (
        f"### 🦴 Predicted Skeletal Maturation:\n"
        f"**{cvm['code']} ({cvm['phase']})** - *Confidence: {cvm['confidence']}%*\n\n"
        f"> {cvm['description']}"
    )

    table_data = []
    for lm in results["landmarks"]:
        table_data.append({
            "No": lm["index"],
            "Symbol": lm["symbol"],
            "Name": lm["name"],
            "X (px)": lm["x"],
            "Y (px)": lm["y"],
            "Confidence": lm["confidence"],
            "Description": lm["description"]
        })

    df = pd.DataFrame(table_data)
    rendered_pil = Image.fromarray(results["rendered_image"])

    return rendered_pil, cvm_markdown, df


# Sample radiographs
sample_dir = Path(__file__).parent / "samples"
sample_files = [[str(f)] for f in sample_dir.glob("*.png")]

with gr.Blocks(title="Oskar Cephalo v2 - AI Landmark Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🦷 Oskar Cephalo v2 — AI Cephalometric Landmark Detection
        ### High-Resolution Automated Detection of 29 Anatomical Landmarks & CVM Stage Classification
        * **Engine**: ONNX Runtime (CPU / GPU Accelerated)
        * **Model**: `oskar_cephalo.onnx` (HRNet Architecture + Adaptive Wing Loss)
        * **Clinical Performance**: **MRE: 0.77 mm** | **SDR@2mm: 95.4%** | **SDR@4mm: 99.5%**
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Lateral Cephalogram X-Ray")

            with gr.Accordion("⚙️ Visualization Options", open=False):
                point_slider = gr.Slider(minimum=2, maximum=12, value=6, step=1, label="Point Size")
                labels_check = gr.Checkbox(value=True, label="Show Landmark Symbols (A, N, B, etc.)")
                lines_check = gr.Checkbox(value=True, label="Draw Reference Planes (S-N, Frankfort, Mandibular)")

            predict_btn = gr.Button("🔍 Predict Landmarks & CVM", variant="primary", size="lg")

            if sample_files:
                gr.Examples(
                    examples=sample_files,
                    inputs=[input_img],
                    label="Sample Test X-Rays (Click to test instantly)"
                )

        with gr.Column(scale=1):
            output_img = gr.Image(type="pil", label="Detection Overlay Result")
            cvm_card = gr.Markdown("### 🦴 Predicted Skeletal Maturation:\n*Upload an image and run prediction.*")

    with gr.Row():
        landmarks_table = gr.Dataframe(
            label="📍 29 Detected Anatomical Landmarks (Coordinates & Descriptions)",
            headers=["No", "Symbol", "Name", "X (px)", "Y (px)", "Confidence", "Description"],
            datatype=["number", "str", "str", "number", "number", "number", "str"],
            interactive=False
        )

    predict_btn.click(
        fn=predict,
        inputs=[input_img, point_slider, labels_check, lines_check],
        outputs=[output_img, cvm_card, landmarks_table]
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
