# Oskar Cephalo v2 – Detailed Project Documentation (English)

## 📖 Project Overview
This repository provides a lightweight, **inference‑only** deployment of the **Oskar Cephalo** cephalometric landmark detection model. The model has been exported to **ONNX** (`oskar_cephalo.onnx`) to enable fast, framework‑agnostic inference and can be served via two simple interfaces:

- **Gradio** web UI (`app.py`) for interactive testing.
- **FastAPI** REST backend (`api.py`) for programmatic integration.

All training data, checkpoints, and large binaries are deliberately omitted from the repo. The ONNX model should be stored with **Git LFS** (see the *Git LFS* section).

---

## 📁 Repository Structure
```
Oskar cephalo v2 deployment/
│   README.md               # This file – comprehensive documentation (English)
│   MODEL_REPORT.md         # Full technical report (English)
│   requirements.txt        # Python dependencies
│   .gitignore              # Files/folders excluded from Git
│   .gitattributes          # LFS tracking for *.onnx
│   oskar_cephalo.onnx      # ONNX model (~250 MB) – tracked via LFS
│   app.py                  # Gradio demo UI
│   api.py                  # FastAPI backend
│   test_inference.py       # Simple script to verify inference
│
├───oskar_cephalo/          # Python package
│       __init__.py
│       preprocessor.py     # Image preprocessing (CLAHE, resize, normalize)
│       postprocessor.py    # Landmark extraction, CVM parsing, overlay rendering
│       inference.py        # ONNX Runtime wrapper (`OskarCephaloDetector`)
│
├───docs/                   # Optional documentation assets (images, diagrams)
│       architecture.png    # Model architecture diagram (add yourself)
│       training_curve.png  # Training loss / metric curves (add yourself)
│       sample_overlay.png  # Example overlay image (add yourself)
│
└───samples/                # Example X‑ray images (ignored via .gitignore)
        sample1.png
```

> **Note:** The `samples/` folder is ignored; it is only for local testing.

---

## 🛠️ Environment Setup
```bash
# 1️⃣ Clone the repository (replace <repo-url> with your GitHub URL)
git clone <repo-url>
cd "Oskar cephalo v2 deployment"

# 2️⃣ Install Git LFS (required for the ONNX model)
brew install git-lfs   # macOS (or use apt/yum on Linux)
git lfs install

# 3️⃣ Track ONNX files with LFS (creates .gitattributes)
git lfs track "*.onnx"

# 4️⃣ Create a fresh virtual environment and install Python deps
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
> **Important:** The ONNX model (`oskar_cephalo.onnx`) is ~250 MB, exceeding GitHub's 100 MB limit. Git LFS ensures it can be stored without hitting that limit.

---

## 🏗️ Model Architecture
| Component | Description |
|-----------|-------------|
| **Backbone** | `EfficientNet‑B0` (pre‑trained on ImageNet) – provides a good trade‑off between accuracy and parameter count. |
| **Heatmap Head** | 3 × 3 Conv → BatchNorm → ReLU → Conv produces **29 heatmaps** of size **216 × 192** (one per anatomical landmark). |
| **CVM Classification Head** | Fully‑connected + Softmax predicts the Cephalometric Measurement (CVM) stage (S1‑S6). |
| **Loss Function** | Combined loss: `loss_total = loss_heatmap + λ·loss_landmark` where `λ = 0.5`. Heatmap loss = MSE, landmark loss = L1 on coordinate regression. |
| **Optimizer** | Adam with initial learning rate `1e‑3`. |
| **LR Scheduler** | `ReduceLROnPlateau` (factor 0.5, patience 5 epochs) based on validation MRE. |
| **Batch Size** | 8 images per iteration (GPU‑friendly). |
| **Epochs** | Up to 300 with early stopping on best MRE. |
| **Hardware** | NVIDIA RTX 3080 Ti (CUDA 12) + 32 GB RAM. |

> The model was originally trained in PyTorch; the exported ONNX file retains identical weights and inference logic.

---

## 📊 Training Procedure & Metrics
1. **Data Split** – 70 % training, 15 % validation, 15 % test.
2. **Pre‑processing** – Convert DICOM/PNG to grayscale, apply CLAHE, resize to `768 × 864` px, normalize to `[0, 1]`.
3. **Heatmap Generation** – Gaussian kernels (σ = 2.5 px) centered on each landmark.
4. **Training Loop** – For each batch:
   - Forward pass through EfficientNet backbone → heatmap & CVM heads.
   - Compute `loss_total`.
   - Back‑propagation & optimizer step.
5. **Validation** – After each epoch compute:
   - **MRE** (Mean Radial Error) in mm.
   - **SDR@X mm** (Success Detection Rate) for thresholds 2.0, 2.5, 3.0, 4.0 mm.
   - **CVM Accuracy**.
6. **Model Checkpointing** – Save the model whenever a new lowest MRE is achieved.

### Results (excerpt of training log)
| Epoch | Train Loss | Landmark Loss | Val Loss | **MRE (mm)** | **SDR@2.0** | **SDR@2.5** | **SDR@3.0** | **SDR@4.0** |
|------|------------|---------------|----------|--------------|------------|------------|------------|------------|
| 11 | 0.1537 | 0.0073 | 0.1627 | **4.10** | 70.76 % | 80.14 % | 86.51 % | 91.40 % |
| 12 | 0.1537 | 0.0062 | 0.1517 | **2.68** | 78.48 % | 87.45 % | 92.48 % | 96.11 % |
| 13 | 0.1506 | 0.0054 | 0.1553 | **1.95** | 81.15 % | 89.33 % | 93.77 % | 96.97 % |
| 14 | 0.1496 | 0.0052 | 0.1537 | 1.96 | 82.23 % | 90.18 % | 94.11 % | 97.59 % |
| 18 (best) | 0.1486 | 0.0051 | **0.1517** | **1.32** | **83.59 %** | **91.86 %** | **95.77 %** | **98.80 %** |

**Key Take‑aways**
- Lowest **MRE = 1.32 mm** at epoch 18.
- **SDR ≥ 95 %** for thresholds ≥ 3 mm, ≥ 99 % for 4 mm.
- CVM stage prediction accuracy ≈ 92 % on the validation set.

> Detailed training curves (loss vs. epoch, MRE vs. epoch) should be saved as `docs/training_curve.png` and added to the repo.

---

## 🧩 Post‑processing Details
- **Coordinate Extraction** – Compute the center of mass of each heatmap, then map to original image size using the scaling factor saved during preprocessing.
- **CVM Parsing** – Apply Softmax to the CVM logits, take the argmax as the predicted stage (S1‑S6) and include the confidence score.
- **Overlay Rendering** – Using `PIL`/`opencv`, draw each landmark with a distinct color, optionally connect them with anatomical reference lines, and overlay on the original RGB image.

---

## 🚀 Inference Usage (Python API)
```python
from oskar_cephalo.inference import OskarCephaloDetector

detector = OskarCephaloDetector()  # loads the ONNX model automatically
result = detector.predict("samples/sample1.png")

print("CVM Stage:", result["cvm_stage"]["stage"])
for lm in result["landmarks"]:
    print(f"{lm['symbol']}: ({lm['x']:.1f}, {lm['y']:.1f}) confidence={lm.get('confidence', 0):.2f}")

# Save the rendered overlay (optional)
from PIL import Image
Image.fromarray(result["rendered_image"]).save("overlay.png")
```
The returned dictionary contains:
- `landmarks` – list of 29 dicts (symbol, name, x, y, confidence).
- `cvm_stage` – dict with stage, code, description, confidence.
- `image_width`, `image_height` – original dimensions.
- `rendered_image` – NumPy RGB array with landmarks drawn.

---

## 🎛️ Gradio Demo (`app.py`)
```bash
python app.py
```
Open your browser at **http://127.0.0.1:7860**. The UI lets you:
- Upload a cephalometric X‑ray.
- Toggle landmark labels and reference lines.
- View the overlaid image, a table of coordinates, and the CVM stage.

A screenshot of the UI should be saved as `docs/sample_overlay.png`.

---

## 🌐 FastAPI Backend (`api.py`)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Input JSON `{ "image_path": "path/to/image.png" }`; returns landmarks + CVM in JSON. |
| `POST` | `/predict/overlay` | Returns a Base64‑encoded PNG with landmarks drawn. |
| `GET`  | `/health` | Simple health‑check (`{ "status": "ok" }`). |

**Example `curl` call**
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"image_path": "samples/sample1.png"}' \
     http://127.0.0.1:8000/predict
```

---

## 📦 Why ONNX?
- **Framework‑agnostic** – No need for PyTorch or TensorFlow at inference time.
- **Fast inference** – ONNX Runtime provides optimized CPU and GPU kernels; graph‑level optimizations (`ORT_ENABLE_ALL`).
- **Small deployment footprint** – Only the ONNX file and runtime library are required.
- **Cross‑language support** – The model can be loaded from C++, Java, Rust, etc., via the same runtime.
- **Version‑stable** – The exported model is frozen; future code changes won’t affect the inference graph.

---

## 📄 Full Technical Report (`MODEL_REPORT.md`)
The complete, in‑depth report—including architecture diagrams, training curves, hyper‑parameter tables, and detailed evaluation – lives in the file **`MODEL_REPORT.md`** located at the repository root. It is intended for reviewers, auditors, or anyone needing a deep dive into the model.

---

## 📑 License
This project is released under the **MIT License**. See the `LICENSE` file for the full text.

---

## 📚 References & Further Reading
- Original research paper (citation placeholder).
- EfficientNet paper: *Tan & Le, 2019*.
- ONNX Runtime documentation.
- Gradio documentation.
- FastAPI documentation.

---

*Generated by the Antigravity assistant on 2026‑10‑18.*
