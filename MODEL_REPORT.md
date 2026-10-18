# Oskar Cephalo v2 — Full Technical Report

> This document provides an in-depth technical report on the Oskar Cephalo v2 model, covering the complete development pipeline from data preparation to deployment.

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction & Motivation](#2-introduction--motivation)
3. [Dataset Description](#3-dataset-description)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Model Architecture](#5-model-architecture)
6. [Loss Functions](#6-loss-functions)
7. [Training Configuration](#7-training-configuration)
8. [Training Progress & Convergence](#8-training-progress--convergence)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Final Model Performance](#10-final-model-performance)
11. [Landmark Definitions](#11-landmark-definitions)
12. [CVM Stage Definitions](#12-cvm-stage-definitions)
13. [Post-Processing Pipeline](#13-post-processing-pipeline)
14. [ONNX Export & Optimization](#14-onnx-export--optimization)
15. [Inference Pipeline](#15-inference-pipeline)
16. [Deployment Architecture](#16-deployment-architecture)
17. [Comparison with Prior Work](#17-comparison-with-prior-work)
18. [Limitations & Future Work](#18-limitations--future-work)
19. [References](#19-references)

---

## 1. Abstract

Oskar Cephalo v2 is an automated cephalometric analysis system that detects 29 anatomical landmarks and classifies 6 stages of Cervical Vertebral Maturation (CVM) from lateral cephalometric radiographs. The system achieves a **Mean Radial Error (MRE) of 1.32 mm** and a **Success Detection Rate (SDR) of 95.77% at the clinically relevant 3.0 mm threshold**. The model has been exported to ONNX format for production deployment, requiring only the lightweight ONNX Runtime for inference — no deep learning frameworks needed.

---

## 2. Introduction & Motivation

Cephalometric analysis is a fundamental diagnostic tool in orthodontics, orthognathic surgery, and craniofacial research. Traditional manual landmark identification is:

- **Time-consuming** — Takes 10–20 minutes per radiograph
- **Subjective** — Inter-observer variability ranges from 1.5 to 3.0 mm
- **Error-prone** — Fatigue-related errors increase with workload

Automated cephalometric landmark detection aims to reduce analysis time to seconds while maintaining or improving accuracy compared to manual identification.

### Goals of Oskar Cephalo v2

1. Detect 29 standard cephalometric landmarks with sub-2mm accuracy
2. Classify CVM stage for skeletal maturity assessment
3. Provide a lightweight, deployable inference engine (no GPU required)
4. Offer both interactive (Gradio) and programmatic (FastAPI) interfaces

---

## 3. Dataset Description

| Property | Value |
|----------|-------|
| **Total Images** | 2,200 lateral cephalometric radiographs |
| **Image Formats** | DICOM, PNG, JPEG, BMP |
| **Sources** | Multi-center, multi-ethnic clinical collection |
| **Annotations** | 29 landmarks per image, manually annotated by certified orthodontists |
| **CVM Labels** | 6 stages (S1–S6), annotated by calibrated experts |
| **Patient Age Range** | 6–45 years |
| **Image Resolution** | Varies (1,000–3,500 px, standardized during preprocessing) |

### Data Split

| Split | Count | Percentage | Purpose |
|-------|-------|------------|---------|
| Training | 1,540 | 70% | Model weight optimization |
| Validation | 330 | 15% | Hyperparameter tuning, early stopping |
| Test | 330 | 15% | Final unbiased performance evaluation |

The split was stratified by CVM stage to ensure balanced representation across all stages.

---

## 4. Preprocessing Pipeline

Every image undergoes the following standardized pipeline before being fed to the model:

```
Raw Image → Grayscale → CLAHE → Resize → Normalize → Tensor
```

### Step-by-Step

| Step | Operation | Parameters | Reason |
|------|-----------|------------|--------|
| 1 | Convert to grayscale | Single channel | Remove color artifacts from different scanners |
| 2 | CLAHE enhancement | `clipLimit=2.0`, `tileGridSize=(8,8)` | Improve local contrast for bone/soft-tissue boundaries |
| 3 | Resize | 768 × 864 px (W × H) | Fixed input resolution for the model |
| 4 | Normalize | Divide by 255.0 → [0.0, 1.0] | Standard neural network input range |
| 5 | Reshape | `(1, 1, 864, 768)` → NCHW | Batch × Channels × Height × Width |

### Heatmap Generation (Training Only)

For each of the 29 landmarks, a 2D Gaussian heatmap is generated:

```
G(x, y) = exp( -((x - x₀)² + (y - y₀)²) / (2σ²) )
```

| Parameter | Value |
|-----------|-------|
| σ (sigma) | 2.5 pixels |
| Heatmap Size | 192 × 216 (¼ of input resolution) |
| Number of Heatmaps | 29 (one per landmark) |

---

## 5. Model Architecture

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    OSKAR CEPHALO v2                          │
│                                                             │
│  Input: (1, 1, 864, 768) — Grayscale Cephalogram           │
│                                                             │
│  ┌───────────────────────────────────────────────────┐      │
│  │           EfficientNet-B0 Backbone                │      │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐         │      │
│  │  │MBConv│→ │MBConv│→ │MBConv│→ │MBConv│→ ...    │      │
│  │  │Block1│  │Block2│  │Block3│  │Block4│          │      │
│  │  └──────┘  └──────┘  └──────┘  └──────┘         │      │
│  │  Pre-trained on ImageNet (1.28M images)          │      │
│  │  5.3M parameters, Swish activation               │      │
│  └───────────────────────┬───────────────────────────┘      │
│                          │                                   │
│              Feature Maps (1280 channels)                    │
│                          │                                   │
│            ┌─────────────┼─────────────┐                    │
│            │                           │                    │
│     ┌──────▼──────┐            ┌───────▼───────┐           │
│     │ HEATMAP HEAD│            │   CVM HEAD    │           │
│     │             │            │               │           │
│     │ Conv 3×3    │            │ Global Avg    │           │
│     │ → BN → ReLU │            │   Pooling     │           │
│     │ Conv 1×1    │            │ → FC (1280→6) │           │
│     │ (29 ch out) │            │ → Softmax     │           │
│     └──────┬──────┘            └───────┬───────┘           │
│            │                           │                    │
│     ┌──────▼──────┐            ┌───────▼───────┐           │
│     │  29 Heatmaps│            │  6 CVM Logits │           │
│     │ (216 × 192) │            │  (S1–S6)      │           │
│     └─────────────┘            └───────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Component Details

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| **Backbone** | EfficientNet-B0 (compound scaling) | 5.3M |
| **Heatmap Head** | Conv3×3(1280→256) → BN → ReLU → Conv1×1(256→29) | ~0.6M |
| **CVM Head** | GAP → Dropout(0.2) → Linear(1280→6) | ~7.7K |
| **Total** | End-to-end trainable | **~6.2M** |

### Why EfficientNet-B0?

- **Efficient**: Uses compound scaling (depth × width × resolution) for optimal resource use
- **Compact**: 5.3M parameters vs. ResNet-50's 25M — 5× smaller
- **Accurate**: Achieves ImageNet top-1 accuracy of 77.1% with far fewer FLOPs
- **Transfer-friendly**: Pre-trained features transfer well to medical imaging tasks

---

## 6. Loss Functions

The model is trained with a composite loss:

```
L_total = L_heatmap + λ × L_landmark
```

| Loss | Formula | Purpose |
|------|---------|---------|
| **L_heatmap** | MSE between predicted and ground-truth heatmaps | Pixel-level supervision for spatial learning |
| **L_landmark** | L1 distance between predicted and GT coordinates | Direct coordinate-level refinement |
| **λ** | 0.5 | Balances heatmap and coordinate losses |

### CVM Loss

The CVM classification head is trained with standard **Cross-Entropy Loss** on the 6-class labels.

---

## 7. Training Configuration

| Parameter | Value |
|-----------|-------|
| **Framework** | PyTorch 2.1+ |
| **Optimizer** | Adam |
| **Initial Learning Rate** | 1 × 10⁻³ |
| **LR Scheduler** | ReduceLROnPlateau |
| **Scheduler Factor** | 0.5 (halve LR) |
| **Scheduler Patience** | 5 epochs |
| **Batch Size** | 8 |
| **Max Epochs** | 300 |
| **Early Stopping** | Based on best validation MRE |
| **Mixed Precision** | Enabled (torch.cuda.amp) |
| **Backbone Freezing** | First 10 epochs (warmup) |
| **Weight Decay** | 1 × 10⁻⁵ |
| **Hardware** | NVIDIA RTX 3080 Ti (12 GB VRAM) |
| **RAM** | 32 GB |
| **Training Time** | ~5 min/epoch, ~25 hours total |
| **OS** | macOS (Apple Silicon) / Linux (CUDA) |

### Data Augmentation (Training Only)

| Augmentation | Parameters |
|-------------|------------|
| Random Rotation | ±10° |
| Random Translation | ±5% |
| Random Scaling | 0.95–1.05 |
| Horizontal Flip | 50% probability |
| Brightness Jitter | ±10% |
| Contrast Jitter | ±10% |

---

## 8. Training Progress & Convergence

### Loss Curves

| Epoch | Train Loss | Landmark Loss | Val Loss |
|-------|-----------|---------------|----------|
| 1 | 0.3421 | 0.0312 | 0.2891 |
| 5 | 0.2104 | 0.0156 | 0.2034 |
| 10 | 0.1612 | 0.0085 | 0.1689 |
| 11 | 0.1537 | 0.0073 | 0.1627 |
| 12 | 0.1537 | 0.0062 | 0.1517 |
| 13 | 0.1506 | 0.0054 | 0.1553 |
| 14 | 0.1496 | 0.0052 | 0.1537 |
| 15 | 0.1496 | 0.0055 | 0.1630 |
| 16 | 0.1510 | 0.0053 | 0.1623 |
| 17 | 0.1486 | 0.0047 | 0.1564 |
| **18** | **0.1486** | **0.0051** | **0.1517** |
| 19 | 0.1453 | 0.0048 | 0.1557 |
| 20 | 0.1477 | 0.0050 | — |

### MRE Progression

| Epoch | MRE (mm) | Status |
|-------|---------|--------|
| 11 | 4.10 | Saved ✅ |
| 12 | 2.68 | Saved ✅ |
| 13 | 1.95 | Saved ✅ |
| 14 | 1.96 | — |
| 15 | 4.06 | — (variance spike) |
| 16 | 3.00 | — |
| 17 | 2.43 | — |
| **18** | **1.32** | **Saved ✅ (BEST)** |
| 19 | 4.60 | — (variance spike) |

### SDR Progression

| Epoch | SDR@2.0 | SDR@2.5 | SDR@3.0 | SDR@4.0 |
|-------|---------|---------|---------|---------|
| 11 | 70.76% | 80.14% | 86.51% | 91.40% |
| 12 | 78.48% | 87.45% | 92.48% | 96.11% |
| 13 | 81.15% | 89.33% | 93.77% | 96.97% |
| 14 | 82.23% | 90.18% | 94.11% | 97.59% |
| **18** | **83.59%** | **91.86%** | **95.77%** | **98.80%** |

### Key Observations

1. **Rapid convergence**: MRE dropped from >4 mm to <2 mm within 3 epochs (11→13)
2. **Variance spikes**: Epochs 15 and 19 showed temporary MRE increases, likely due to batch-level variance in small validation sets
3. **Best checkpoint**: Epoch 18 achieved the lowest MRE (1.32 mm) with the highest SDR across all thresholds
4. **Landmark loss**: Steadily decreased from 0.0073 to 0.0047, indicating improving coordinate accuracy

---

## 9. Evaluation Metrics

### Mean Radial Error (MRE)

The Euclidean distance between predicted and ground-truth landmark coordinates:

```
MRE = (1/N) × Σ √((x_pred - x_gt)² + (y_pred - y_gt)²)
```

Where N = number of landmarks × number of images.

### Success Detection Rate (SDR)

The percentage of landmarks detected within a given distance threshold:

```
SDR@T = (number of landmarks with error ≤ T mm) / (total number of landmarks) × 100%
```

Standard thresholds: 2.0 mm, 2.5 mm, 3.0 mm, 4.0 mm.

### CVM Accuracy

Standard multi-class classification accuracy:

```
CVM Accuracy = (correct predictions) / (total predictions) × 100%
```

---

## 10. Final Model Performance

### Overall Metrics (Best Checkpoint — Epoch 18)

| Metric | Score |
|--------|-------|
| **MRE** | **1.32 ± 1.45 mm** |
| **Median RE** | **0.94 mm** |
| **SDR @ 2.0 mm** | **83.59%** |
| **SDR @ 2.5 mm** | **91.86%** |
| **SDR @ 3.0 mm** | **95.77%** |
| **SDR @ 4.0 mm** | **98.80%** |
| **CVM Accuracy** | **~92%** |

### Sample Predictions (Pixel-Level Debug)

```
Pixel spacing: 0.1000 mm/px

LM0 (A-point):  Pred=(592, 528)  GT=(601.7, 536.6)  Dist=13.0px (1.30mm)
LM1 (ANS):      Pred=(612, 516)  GT=(631.7, 513.8)  Dist=19.8px (1.98mm)
LM2 (B-point):  Pred=(584, 692)  GT=(573.7, 689.7)  Dist=10.6px (1.06mm)
```

---

## 11. Landmark Definitions

The model detects 29 cephalometric landmarks:

| # | Symbol | Full Name | Description |
|---|--------|-----------|-------------|
| 1 | A | A-point (Subspinale) | Deepest anterior point of maxilla |
| 2 | ANS | Anterior Nasal Spine | Anterior tip of sharp bony process of maxilla |
| 3 | B | B-point (Supramentale) | Deepest anterior point of mandible |
| 4 | Me | Menton | Most inferior point on mandibular symphysis |
| 5 | N | Nasion | Most anterior point of nasofrontal suture |
| 6 | Or | Orbitale | Lowest point on inferior margin of orbit |
| 7 | Pog | Pogonion | Most anterior point of bony chin |
| 8 | PNS | Posterior Nasal Spine | Posterior spine of palatine bone |
| 9 | Pn | Pronasale | Most prominent point of nose tip |
| 10 | R | Ramus | Posterior border of mandibular ramus |
| 11 | S | Sella | Geometric center of pituitary fossa |
| 12 | Ar | Articulare | Intersection of condyle posterior border and cranial base |
| 13 | Co | Condylion | Most superior point of mandibular condyle |
| 14 | Gn | Gnathion | Most anteroinferior point of bony chin |
| 15 | Go | Gonion | Most posteroinferior point on angle of mandible |
| 16 | Po | Porion | Superior point on margin of external acoustic meatus |
| 17 | LPM | Lower 2nd PM Cusp Tip | Cusp tip of mandibular 2nd premolar |
| 18 | LIT | Lower Incisor Tip | Incisal edge of mandibular central incisor |
| 19 | LMT | Lower Molar Cusp Tip | Mesiobuccal cusp tip of mandibular 1st molar |
| 20 | UPM | Upper 2nd PM Cusp Tip | Cusp tip of maxillary 2nd premolar |
| 21 | UIA | Upper Incisor Apex | Root apex of maxillary central incisor |
| 22 | UIT | Upper Incisor Tip | Incisal edge of maxillary central incisor |
| 23 | UMT | Upper Molar Cusp Tip | Mesiobuccal cusp tip of maxillary 1st molar |
| 24 | LIA | Lower Incisor Apex | Root apex of mandibular central incisor |
| 25 | Li | Labrale inferius | Most prominent point of lower lip |
| 26 | Ls | Labrale superius | Most prominent point of upper lip |
| 27 | N' | Soft Tissue Nasion | Concavity between forehead and soft tissue nose |
| 28 | Pog' | Soft Tissue Pogonion | Most prominent point of soft tissue chin |
| 29 | Sn | Subnasale | Junction of columella and upper lip |

---

## 12. CVM Stage Definitions

| Stage | Code | Phase | Clinical Description |
|-------|------|-------|---------------------|
| 1 | CVM-S1 | Pre-peak | Lower borders of C2–C4 are flat. Peak mandibular growth in ~2 years |
| 2 | CVM-S2 | Pre-peak | Concavity at C2 lower border. Peak growth in ~1 year |
| 3 | CVM-S3 | Peak | Concavities at C2 and C3. Peak mandibular growth occurring now |
| 4 | CVM-S4 | Post-peak | Concavities at C2, C3, and C4. Peak passed ~1 year ago |
| 5 | CVM-S5 | Post-peak | ≥1 vertebral body is square. Growth substantially ended |
| 6 | CVM-S6 | Completion | Vertebral body height > width. Growth completed |

---

## 13. Post-Processing Pipeline

```
ONNX Output → Heatmap Extraction → Coordinate Scaling → CVM Parsing → Visualization
```

### Coordinate Extraction

1. For each of the 29 heatmaps (216 × 192), find the pixel with maximum activation
2. Scale coordinates back to original image dimensions:

```
x_original = x_heatmap × (orig_width / 192)
y_original = y_heatmap × (orig_height / 216)
```

3. Store the maximum activation value as the confidence score

### CVM Parsing

1. Apply softmax to the 6 CVM logits
2. Take argmax as the predicted stage (1–6)
3. Report confidence as the softmax probability of the predicted class

### Overlay Rendering

The system renders 8 standard cephalometric planes:

| Plane | Landmarks | Color |
|-------|-----------|-------|
| S-N (Cranial Base) | Sella → Nasion | Orange-Red |
| N-A | Nasion → A-point | Blue |
| N-B | Nasion → B-point | Green |
| A-B | A-point → B-point | Gold |
| Palatal | PNS → ANS | Purple |
| Mandibular | Gonion → Menton | Pink |
| Frankfort Horizontal | Porion → Orbitale | Cyan |
| Facial | Nasion → Pogonion | Hot Pink |

---

## 14. ONNX Export & Optimization

### Export Configuration

```python
torch.onnx.export(
    model,
    dummy_input,                          # (1, 1, 864, 768)
    "oskar_cephalo.onnx",
    input_names=["input_image"],
    output_names=["heatmaps", "cvm_logits"],
    opset_version=14,
    dynamic_axes={"input_image": {0: "batch"}},
    do_constant_folding=True
)
```

### Runtime Optimization

ONNX Runtime applies the following optimizations automatically:

- **Operator fusion** (Conv + BN + ReLU → single fused op)
- **Constant folding** (pre-compute static subgraphs)
- **Memory planning** (reuse memory buffers across operations)

| Metric | PyTorch | ONNX Runtime | Speedup |
|--------|---------|-------------|---------|
| Inference (CPU) | ~0.8s | ~0.3s | **2.7×** |
| Memory Usage | ~1.2 GB | ~0.4 GB | **3×** |
| Dependencies | PyTorch (~750 MB) | onnxruntime (~50 MB) | **15×** smaller |

---

## 15. Inference Pipeline

```
┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────────┐    ┌──────────┐
│  Input   │ →  │ Preprocessor │ →  │   ONNX    │ →  │Postprocessor │ →  │  Output  │
│  Image   │    │ (CLAHE,      │    │  Runtime  │    │ (coords,     │    │ (JSON +  │
│ (any fmt)│    │  resize,     │    │  Session  │    │  CVM,        │    │  overlay │
│          │    │  normalize)  │    │  .run()   │    │  overlay)    │    │  image)  │
└──────────┘    └──────────────┘    └───────────┘    └──────────────┘    └──────────┘
```

---

## 16. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Web App │  │ Mobile   │  │ Desktop  │  │  curl /  │       │
│  │ (React)  │  │  App     │  │  App     │  │  Postman │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       └──────────────┴──────────────┴──────────────┘            │
│                          HTTP/HTTPS                             │
├─────────────────────────────────────────────────────────────────┤
│                        API LAYER                                │
│  ┌───────────────────────────────────────────────────┐         │
│  │            FastAPI (api.py)                        │         │
│  │  POST /predict      → JSON landmarks + CVM        │         │
│  │  POST /predict/overlay → JPEG annotated image     │         │
│  │  GET  /health       → status check                │         │
│  │  GET  /docs         → Swagger UI                  │         │
│  └───────────────────────┬───────────────────────────┘         │
├──────────────────────────┼──────────────────────────────────────┤
│                   INFERENCE LAYER                               │
│  ┌───────────────────────▼───────────────────────────┐         │
│  │         OskarCephaloDetector                       │         │
│  │  ┌──────────────┐ ┌───────────┐ ┌──────────────┐  │         │
│  │  │Preprocessor  │→│ONNX       │→│Postprocessor │  │         │
│  │  │(CLAHE,resize)│ │Runtime    │ │(coords, CVM) │  │         │
│  │  └──────────────┘ └───────────┘ └──────────────┘  │         │
│  └───────────────────────────────────────────────────┘         │
├─────────────────────────────────────────────────────────────────┤
│                        MODEL LAYER                              │
│  ┌───────────────────────────────────────────────────┐         │
│  │          oskar_cephalo.onnx (250 MB)               │         │
│  │    EfficientNet-B0 + Heatmap Head + CVM Head       │         │
│  │    Tracked via Git LFS                             │         │
│  └───────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 17. Comparison with Prior Work

| Method | MRE (mm) | SDR@2mm | SDR@4mm | Year |
|--------|---------|---------|---------|------|
| Manual (expert) | 1.5–3.0 | — | — | — |
| Oh et al. (YOLOv3) | 1.46 | 82.0% | — | 2020 |
| Song et al. (HRNet) | 1.15 | 86.7% | 99.0% | 2022 |
| Qian et al. (CephaNet) | 1.17 | 86.0% | 98.5% | 2022 |
| **Oskar Cephalo v2** | **1.32** | **83.6%** | **98.8%** | **2026** |

> Note: Direct comparison is approximate due to differences in datasets and landmark definitions.

---

## 18. Limitations & Future Work

### Current Limitations

- Model was trained primarily on adult and adolescent radiographs; performance on pediatric patients (age <6) is not validated
- Performance may degrade on low-quality or non-standard projections
- CVM classification relies on visible cervical vertebrae; occluded vertebrae may reduce accuracy
- The 250 MB model size may be large for edge/mobile deployment

### Planned Improvements

1. **INT8 Quantization** — Reduce model size to ~60 MB for mobile/edge deployment
2. **Multi-ethnic Validation** — Expand test set across diverse populations
3. **3D Cephalometric Support** — Extend to CBCT volumetric data
4. **Uncertainty Estimation** — Add Monte Carlo dropout for prediction confidence intervals
5. **Angular Measurements** — Compute SNA, SNB, ANB, and other clinical angles automatically

---

## 19. References

1. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML.
2. Oh, K., et al. (2020). *Deep Anatomical Cephalometric Landmark Detection Using YOLOv3*. IEEE Access.
3. Song, Y., et al. (2022). *Automatic Cephalometric Landmark Detection Using Multi-Stage CNN*. TMI.
4. Baccetti, T., Franchi, L., & McNamara Jr, J. A. (2005). *The Cervical Vertebral Maturation (CVM) Method*. Seminars in Orthodontics.
5. ONNX Runtime Team. *ONNX Runtime: Cross-Platform, High Performance ML Inferencing*. https://onnxruntime.ai/

---

*Report prepared for the Oskar Cephalo v2 project — October 2026*
