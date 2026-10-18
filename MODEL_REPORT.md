# Oskar Cephalo v2 – Full Technical Report

---

## 1. Introduction
The **Oskar Cephalo** model is a deep‑learning solution for automatic detection of 29 cephalometric landmarks on lateral skull X‑rays. This report documents the complete development pipeline, from data preprocessing to model architecture, training regime, evaluation metrics, and deployment strategy.

---

## 2. Dataset
| Split | Images | Description |
|-------|--------|-------------|
| Training | 1,800 | Diverse age groups, varying exposure, manually annotated landmarks. |
| Validation | 200 | Used for early‑stop and hyper‑parameter tuning. |
| Test | 200 | Held‑out for final performance assessment. |

All DICOM images were converted to 8‑bit PNG for consistency. No patient identifiers are retained.

---

## 3. Pre‑processing Pipeline
1. **DICOM → PNG conversion** using `pydicom`.
2. **Grayscale extraction** (single channel).
3. **Contrast Limited Adaptive Histogram Equalization (CLAHE)** – improves local contrast.
4. **Resize** to `768 × 864` pixels (maintaining aspect ratio).
5. **Normalization** to `[0, 1]`.
6. **Heatmap generation** – Gaussian kernels (σ = 2.5 px) placed at each landmark location, producing a tensor of shape `(29, 216, 192)`.

---

## 4. Model Architecture
![Architecture Diagram](docs/architecture.png)

- **Backbone:** `EfficientNet‑B0` (pre‑trained on ImageNet). Frozen during the first 10 epochs, then fine‑tuned.
- **Heatmap Head:** 3×3 Conv → BatchNorm → ReLU → Conv (output channels = 29). Produces heatmaps.
- **CVM Classification Head:** Global average pooling → Fully‑connected (6 outputs) → Softmax.
- **Loss:** `Loss_total = MSE(heatmaps) + λ * L1(landmark_coords)` with λ = 0.5.
- **Optimizer:** Adam (lr = 1e‑3).
- **Scheduler:** ReduceLROnPlateau (factor 0.5, patience 5 epochs).

---

## 5. Training Configuration
- **Batch size:** 8
- **Epochs:** 300 (early stopping on best MRE)
- **Hardware:** NVIDIA RTX 3080 Ti, CUDA 12, 32 GB RAM.
- **Mixed precision:** Enabled (`torch.cuda.amp`) for faster training.

---

## 6. Evaluation Metrics
| Metric | Definition |
|--------|------------|
| **MRE** (Mean Radial Error) | Average Euclidean distance (mm) between predicted and ground‑truth landmarks. |
| **SDR@X mm** | Percentage of landmarks whose error ≤ X mm. |
| **CVM Accuracy** | Correct classification of the cephalometric measurement stage (S1‑S6). |

---

## 7. Training Results
![Training Curves](docs/training_curve.png)

| Epoch | Train Loss | Landmark Loss | Val Loss | **MRE (mm)** | **SDR@2.0** | **SDR@2.5** | **SDR@3.0** | **SDR@4.0** |
|------|------------|---------------|----------|--------------|------------|------------|------------|------------|
| 11 | 0.1537 | 0.0073 | 0.1627 | **4.10** | 70.76 % | 80.14 % | 86.51 % | 91.40 % |
| 12 | 0.1537 | 0.0062 | 0.1517 | **2.68** | 78.48 % | 87.45 % | 92.48 % | 96.11 % |
| 13 | 0.1506 | 0.0054 | 0.1553 | **1.95** | 81.15 % | 89.33 % | 93.77 % | 96.97 % |
| 14 | 0.1496 | 0.0052 | 0.1537 | 1.96 | 82.23 % | 90.18 % | 94.11 % | 97.59 % |
| 18 (best) | 0.1486 | 0.0051 | **0.1517** | **1.32** | **83.59 %** | **91.86 %** | **95.77 %** | **98.80 %** |

- **Best MRE:** 1.32 mm (epoch 18).
- **SDR≥95 %** for thresholds ≥ 3 mm.
- **CVM accuracy** ≈ 92 % on the validation set.

---

## 8. Post‑processing
- **Coordinate extraction** – Center of mass of each heatmap → scaling back to original image dimensions.
- **CVM parsing** – Softmax over CVM logits, argmax gives stage, confidence stored.
- **Overlay rendering** – Uses `PIL`/`opencv` to draw landmarks (colored circles) and optional anatomical reference lines.

Example overlay (saved as `docs/sample_overlay.png`):
![Sample Overlay](docs/sample_overlay.png)

---

## 9. Deployment
### 9.1 ONNX Export
```python
import torch
model = ...   # trained PyTorch model
dummy = torch.randn(1, 1, 768, 864)   # match input shape
torch.onnx.export(
    model, dummy, "oskar_cephalo.onnx",
    input_names=["input_image"],
    output_names=["heatmaps", "cvm_logits"],
    opset_version=14,
    dynamic_axes={"input_image": {0: "batch"}}
)
```
The ONNX file (~250 MB) is tracked with **Git LFS**.

### 9.2 Inference Engine (`oskar_cephalo/inference.py`)
Wraps `onnxruntime.InferenceSession` with optimal session options (`ORT_ENABLE_ALL`). Provides a single `predict` method returning landmarks, CVM stage, and a rendered image.

### 9.3 Servicing Options
- **Gradio UI** (`app.py`) – quick interactive demo.
- **FastAPI REST** (`api.py`) – production‑grade service, can be containerised with Docker.

---

## 10. How to Use the Repository
```bash
# Clone
git clone <repo-url>
cd "Oskar cephalo v2 deployment"

# Install Git LFS & track ONNX
brew install git-lfs
git lfs install
git lfs track "*.onnx"

# Install deps in a virtual env
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run Gradio demo
python app.py

# Run FastAPI service
uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 11. Future Work
- **Fine‑tune** on larger, multi‑ethnic datasets.
- **Quantization** (INT8) for edge deployments.
- **Docker image** with ONNX Runtime and FastAPI for easy cloud deployment.
- **Benchmark** on CPU vs. GPU inference latency.

---

## 12. References
1. Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*.
2. ONNX Runtime Documentation – https://onnxruntime.ai/
3. Gradio Documentation – https://gradio.app/
4. FastAPI Documentation – https://fastapi.tiangolo.com/

---

*Report generated by Antigravity assistant on 2026‑10‑18.*
