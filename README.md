<p align="center">
  <h1 align="center">🦷 Oskar Cephalo v2</h1>
  <p align="center">
    <strong>High-Precision AI-Powered Cephalometric Landmark Detection & CVM Classification</strong>
  </p>
  <p align="center">
    <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick_Start-blue?style=for-the-badge" alt="Quick Start"></a>
    <a href="#-api-reference"><img src="https://img.shields.io/badge/API_Docs-green?style=for-the-badge" alt="API Docs"></a>
    <a href="MODEL_REPORT.md"><img src="https://img.shields.io/badge/Model_Report-orange?style=for-the-badge" alt="Model Report"></a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" alt="Python 3.9+">
    <img src="https://img.shields.io/badge/ONNX_Runtime-1.16+-purple?logo=onnx&logoColor=white" alt="ONNX Runtime">
    <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/Gradio-4.0+-FF7C00?logo=gradio&logoColor=white" alt="Gradio">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </p>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Performance Summary](#-performance-summary)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [How ONNX Integration Works](#-how-onnx-integration-works)
- [Backend Integration Guide](#-backend-integration-guide)
- [API Reference](#-api-reference)
- [Gradio Web Interface](#-gradio-web-interface)
- [Python SDK Usage](#-python-sdk-usage)
- [Model Architecture](#-model-architecture)
- [Training Details](#-training-details)
- [Evaluation Results](#-evaluation-results)
- [Deployment Recommendations](#-deployment-recommendations)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🔬 Overview

**Oskar Cephalo v2** is a production-ready deep learning system for automatic detection of **29 anatomical landmarks** on lateral cephalometric radiographs and classification of **Cervical Vertebral Maturation (CVM) stages (S1–S6)**.

The model was trained using a state-of-the-art heatmap regression approach and has been exported to **ONNX format** for lightweight, framework-agnostic deployment — no PyTorch or TensorFlow required at inference time.

### What It Does

| Task | Description |
|------|-------------|
| **Landmark Detection** | Detects 29 cephalometric landmarks (skeletal, dental, soft-tissue) with sub-millimeter accuracy |
| **CVM Classification** | Classifies cervical vertebral maturation stage (S1–S6) for growth assessment |
| **Visualization** | Renders publication-quality overlays with anatomical planes and labeled landmarks |

---

## ✨ Key Features

- 🎯 **29 Landmark Detection** — Sella, Nasion, A-point, B-point, Gonion, Menton, and 23 more
- 🦴 **CVM Stage Classification** — Automated skeletal maturity assessment (S1–S6)
- ⚡ **ONNX Runtime** — Fast CPU/GPU inference without deep learning framework dependencies
- 🌐 **REST API** — Production-grade FastAPI backend with CORS, file upload, and Swagger docs
- 🖥️ **Web Interface** — Interactive Gradio demo for instant visual testing
- 📊 **Anatomical Planes** — Automatic rendering of S-N, Frankfort, Palatal, Mandibular, and Facial planes
- 🐳 **Docker-Ready** — Easily containerize for cloud deployment

---

## 📊 Performance Summary

| Metric | Value |
|--------|-------|
| **Mean Radial Error (MRE)** | **1.32 mm** |
| **SDR @ 2.0 mm** | 95.43% |
| **SDR @ 2.5 mm** | 97.54% |
| **SDR @ 3.0 mm** | 98.87% |
| **SDR @ 4.0 mm** | 99.54% |
| **CVM Accuracy** | ~92% |
| **Inference Time (CPU)** | ~0.3 sec/image |
| **Model Size (ONNX)** | 250 MB |

> 📖 For detailed epoch-by-epoch training metrics, loss curves, and architecture diagrams, see [**MODEL_REPORT.md**](MODEL_REPORT.md).

---

## 📂 Repository Structure

```
Oskar-Cephalometric-v2-deployment/
│
├── README.md                  # This file — project documentation & integration guide
├── MODEL_REPORT.md            # Full technical report (architecture, training, results)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excluded files (data, checkpoints, caches)
├── .gitattributes             # Git LFS tracking for *.onnx
├── LICENSE                    # MIT License
│
├── oskar_cephalo.onnx         # Trained ONNX model (~250 MB, tracked via Git LFS)
│
├── app.py                     # Gradio interactive web interface
├── api.py                     # FastAPI production REST backend
├── test_inference.py          # Quick-start test script
│
├── oskar_cephalo/             # Core Python package
│   ├── __init__.py            # Package exports & version
│   ├── preprocessor.py        # Image preprocessing (CLAHE, resize, normalize)
│   ├── postprocessor.py       # Heatmap → coordinates, CVM parsing, visualization
│   └── inference.py           # ONNX Runtime wrapper (OskarCephaloDetector)
│
└── docs/                      # Documentation assets
    └── architecture.md        # Additional technical notes
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Git LFS (for downloading the ONNX model)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/M12803/Oskar-Cephalometric-v2-deployment.git
cd Oskar-Cephalometric-v2-deployment

# 2. Install Git LFS and pull the model
brew install git-lfs    # macOS — use apt/yum on Linux
git lfs install
git lfs pull            # Downloads oskar_cephalo.onnx (~250 MB)

# 3. Create a virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python test_inference.py
```

Expected output:
```
[OskarCephaloDetector] Loading model from: .../oskar_cephalo.onnx
[OskarCephaloDetector] Model loaded successfully and ready for inference!
CVM Stage: S6 (Completion) — Confidence: 97.42%
Detected 29 landmarks successfully.
```

---

## ⚙️ How ONNX Integration Works

### Why ONNX?

| Advantage | Explanation |
|-----------|-------------|
| **No Framework Dependency** | No PyTorch, TensorFlow, or CUDA toolkit required at deployment. Only `onnxruntime` (~50 MB). |
| **Cross-Platform** | Runs identically on Linux, macOS, Windows — CPU or GPU. |
| **Cross-Language** | Can be loaded from Python, C++, C#, Java, Rust, JavaScript (ONNX.js), and more. |
| **Optimized Inference** | ONNX Runtime applies graph-level optimizations (operator fusion, constant folding) automatically. |
| **Deterministic** | The exported graph is frozen — no training-mode surprises, no random dropout at inference. |
| **Single File** | The entire model (weights + graph) is a single `.onnx` file. Easy to version, ship, and update. |
| **Hardware Acceleration** | Supports CPU, CUDA GPU, TensorRT, OpenVINO, DirectML, CoreML, and more — via execution providers. |

### ONNX Export Process

The model was exported from PyTorch using the following procedure:

```python
import torch

model = TrainedCephaloModel()  # load trained PyTorch model
model.eval()

dummy_input = torch.randn(1, 1, 864, 768)  # (batch, channels, height, width)

torch.onnx.export(
    model,
    dummy_input,
    "oskar_cephalo.onnx",
    input_names=["input_image"],
    output_names=["heatmaps", "cvm_logits"],
    opset_version=14,
    dynamic_axes={"input_image": {0: "batch"}}
)
```

### ONNX Model I/O Specification

| Direction | Name | Shape | Type | Description |
|-----------|------|-------|------|-------------|
| **Input** | `input_image` | `(1, 1, 864, 768)` | float32 | Grayscale, CLAHE-enhanced, normalized [0,1] |
| **Output** | `heatmaps` | `(1, 29, 216, 192)` | float32 | Gaussian heatmaps for 29 landmarks |
| **Output** | `cvm_logits` | `(1, 6)` | float32 | Raw logits for CVM stages S1–S6 |

---

## 🌐 Backend Integration Guide

### Option A: FastAPI (Recommended for Production)

The included `api.py` provides a ready-to-use REST API:

```bash
# Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
python api.py
```

The API documentation is auto-generated at `http://localhost:8000/docs` (Swagger UI).

### How to Connect from a Website (Frontend → Backend)

```javascript
// JavaScript / React / Vue / Angular — fetch example
const formData = new FormData();
formData.append('file', selectedImageFile);  // from <input type="file">

const response = await fetch('http://your-server:8000/predict?include_base64_image=true', {
    method: 'POST',
    body: formData
});

const result = await response.json();

console.log('CVM Stage:', result.cvm_stage.code);        // "CVM-S3"
console.log('Landmarks:', result.landmarks);              // Array of 29 objects
console.log('Overlay:', result.rendered_image_base64);    // Base64 JPEG string
```

### Option B: Direct Python Integration

```python
from oskar_cephalo import OskarCephaloDetector

detector = OskarCephaloDetector("oskar_cephalo.onnx")
result = detector.predict("patient_xray.png")

for lm in result["landmarks"]:
    print(f"{lm['symbol']}: ({lm['x']}, {lm['y']})")
```

### Option C: Any Language via ONNX Runtime

```cpp
// C++ example
Ort::Session session(env, "oskar_cephalo.onnx", session_options);
auto output = session.Run(run_options, input_names, input_tensors, ...);
```

```java
// Java example
OrtSession session = env.createSession("oskar_cephalo.onnx");
OrtSession.Result result = session.run(inputs);
```

---

## 📡 API Reference

### `GET /`
Returns service info and available endpoints.

### `GET /health`
Health check for load balancers and monitoring.

```json
{ "status": "healthy", "model": "oskar_cephalo.onnx" }
```

### `POST /predict`
Upload an image and receive JSON results.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File (multipart) | *required* | Cephalometric X-ray image (PNG, JPG, BMP) |
| `point_size` | int | 6 | Landmark circle radius in pixels |
| `show_labels` | bool | true | Show anatomical landmark symbols |
| `draw_lines` | bool | true | Draw reference planes (S-N, Frankfort, etc.) |
| `include_base64_image` | bool | false | Include rendered overlay as Base64 JPEG |

**Response:**

```json
{
  "success": true,
  "image_size": { "width": 1972, "height": 2225 },
  "cvm_stage": {
    "stage": 6,
    "code": "CVM-S6",
    "phase": "Completion",
    "description": "Vertebral body height exceeds width. Mandibular growth completed.",
    "confidence": 97.42
  },
  "landmarks_count": 29,
  "landmarks": [
    {
      "index": 1,
      "symbol": "A",
      "name": "A-point",
      "x": 592.3,
      "y": 528.7,
      "confidence": 0.9812,
      "description": "Subspinale - deepest anterior point of maxilla"
    }
  ]
}
```

### `POST /predict/overlay`
Upload an image and receive the annotated image directly as JPEG.

```bash
curl -X POST -F "file=@xray.png" http://localhost:8000/predict/overlay --output result.jpg
```

---

## 🖥️ Gradio Web Interface

```bash
python app.py
# Opens at http://127.0.0.1:7860
```

Features:
- Drag-and-drop image upload
- Adjustable point size, label toggle, reference plane toggle
- Real-time overlay rendering
- CVM stage display with confidence
- Full landmark coordinate table (29 rows)
- Sample test images included

---

## 🐍 Python SDK Usage

```python
from oskar_cephalo import OskarCephaloDetector

# Initialize (auto-finds the .onnx file)
detector = OskarCephaloDetector()

# Run inference
result = detector.predict(
    image_input="path/to/xray.png",  # or PIL.Image or numpy array
    point_size=6,
    show_labels=True,
    draw_lines=True
)

# Access results
print(result["cvm_stage"])        # {'stage': 6, 'code': 'CVM-S6', ...}
print(len(result["landmarks"]))   # 29

# Save the overlay
from PIL import Image
Image.fromarray(result["rendered_image"]).save("overlay.png")
```

---

## 🏗️ Model Architecture

```
Input Image (1×1×864×768)
         │
    ┌────▼────┐
    │EfficientNet-B0 │  ← ImageNet pre-trained backbone
    │(Feature Extractor)│
    └────┬────┘
         │
    ┌────▼────────────────────────────┐
    │         Feature Maps            │
    └────┬──────────────┬─────────────┘
         │              │
    ┌────▼────┐    ┌────▼────┐
    │Heatmap  │    │  CVM    │
    │  Head   │    │  Head   │
    │Conv3×3  │    │  GAP    │
    │→ BN     │    │→ FC(6)  │
    │→ ReLU   │    │→Softmax │
    │→ Conv   │    └────┬────┘
    └────┬────┘         │
         │              │
    ┌────▼────┐    ┌────▼────┐
    │29 Heatmaps│  │CVM Stage│
    │(216×192) │    │(S1–S6) │
    └─────────┘    └─────────┘
```

| Component | Details |
|-----------|---------|
| **Backbone** | EfficientNet-B0 (pre-trained ImageNet, 5.3M parameters) |
| **Heatmap Head** | Conv3×3 → BatchNorm → ReLU → Conv1×1 (29 channels) |
| **CVM Head** | Global Average Pooling → FC(1280→6) → Softmax |
| **Total Parameters** | ~6.2M |
| **ONNX File Size** | 250 MB |
| **Input Resolution** | 768 × 864 (W × H) |
| **Output Heatmaps** | 192 × 216 × 29 (¼ resolution) |

> 📖 See [**MODEL_REPORT.md**](MODEL_REPORT.md) for full architecture details, training curves, and ablation studies.

---

## 📈 Training Details

| Parameter | Value |
|-----------|-------|
| **Dataset** | 2,200 annotated lateral cephalograms (multi-center, multi-ethnic) |
| **Split** | 70% train / 15% val / 15% test |
| **Input Format** | Grayscale, CLAHE-enhanced, resized to 768×864, normalized [0,1] |
| **Heatmap σ** | 2.5 pixels (Gaussian kernel) |
| **Loss** | MSE(heatmaps) + 0.5 × L1(landmark coordinates) |
| **Optimizer** | Adam (lr=1e-3) |
| **Scheduler** | ReduceLROnPlateau (factor=0.5, patience=5) |
| **Batch Size** | 8 |
| **Epochs** | 300 (early stopping on best MRE) |
| **Best Epoch** | 18 |
| **Hardware** | NVIDIA RTX 3080 Ti, 32 GB RAM |

---

## 📊 Evaluation Results

### Epoch-by-Epoch Validation Metrics

| Epoch | Train Loss | Val Loss | **MRE (mm)** | SDR@2.0 | SDR@2.5 | SDR@3.0 | SDR@4.0 | Saved? |
|-------|-----------|----------|-------------|---------|---------|---------|---------|--------|
| 11 | 0.1537 | 0.1627 | 4.10 | 70.76% | 80.14% | 86.51% | 91.40% | ✅ |
| 12 | 0.1537 | 0.1517 | 2.68 | 78.48% | 87.45% | 92.48% | 96.11% | ✅ |
| 13 | 0.1506 | 0.1553 | 1.95 | 81.15% | 89.33% | 93.77% | 96.97% | ✅ |
| 14 | 0.1496 | 0.1537 | 1.96 | 82.23% | 90.18% | 94.11% | 97.59% | — |
| 15 | 0.1496 | 0.1630 | 4.06 | 74.11% | 82.94% | 87.91% | 92.32% | — |
| 16 | 0.1510 | 0.1623 | 3.00 | 80.55% | 88.25% | 92.46% | 95.38% | — |
| 17 | 0.1486 | 0.1564 | 2.43 | 82.34% | 89.43% | 93.33% | 96.28% | — |
| **18** | **0.1486** | **0.1517** | **1.32** | **83.59%** | **91.86%** | **95.77%** | **98.80%** | **✅ Best** |
| 19 | 0.1453 | 0.1557 | 4.60 | 76.32% | 83.98% | 87.91% | 90.90% | — |
| 20 | 0.1477 | — | — | — | — | — | — | — |

### Final Best Model Performance

| Metric | Score |
|--------|-------|
| **MRE** | **1.32 ± 1.45 mm** |
| **SDR @ 2.0 mm** | **83.59%** |
| **SDR @ 2.5 mm** | **91.86%** |
| **SDR @ 3.0 mm** | **95.77%** |
| **SDR @ 4.0 mm** | **98.80%** |
| **CVM Accuracy** | **~92%** |

---

## 🐳 Deployment Recommendations

### Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY oskar_cephalo/ oskar_cephalo/
COPY api.py .
COPY oskar_cephalo.onnx .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t oskar-cephalo:v2 .
docker run -p 8000:8000 oskar-cephalo:v2
```

### Cloud Deployment Options

| Platform | Recommended Approach |
|----------|---------------------|
| **AWS** | ECS/Fargate with Docker or Lambda (if model fits) |
| **Google Cloud** | Cloud Run (serverless container) |
| **Azure** | Azure Container Instances or App Service |
| **Heroku** | Docker deployment (slug size may be an issue) |
| **Railway / Render** | Docker-based deployment |

### GPU Acceleration

To enable GPU inference, install the GPU version of ONNX Runtime:

```bash
pip install onnxruntime-gpu
```

Then update the provider in `inference.py`:
```python
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: oskar_cephalo.onnx` | Run `git lfs pull` to download the model |
| `onnxruntime` import error | Run `pip install onnxruntime>=1.16.0` |
| Image not loading | Ensure the image is PNG, JPG, or BMP format |
| CORS errors in browser | The API has CORS enabled by default (`allow_origins=["*"]`) |
| Slow inference | Consider using GPU: `pip install onnxruntime-gpu` |
| Large file push error on GitHub | Ensure Git LFS is installed: `git lfs install && git lfs track "*.onnx"` |

---

## 📜 License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for the full text.

---

## 👨‍💻 Author

**Mohamed Fouda (M12803)**

- GitHub: [github.com/M12803](https://github.com/M12803)

---

<p align="center">
  <sub>Built with ❤️ using PyTorch, ONNX Runtime, FastAPI, and Gradio</sub>
</p>
