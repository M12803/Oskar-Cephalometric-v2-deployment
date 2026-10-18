# Oskar Cephalo v2 — Architecture Notes

This document provides supplementary technical notes about the model architecture.

## Input/Output Specification

```
Input:  input_image   — float32, shape (1, 1, 864, 768)
Output: heatmaps      — float32, shape (1, 29, 216, 192)
Output: cvm_logits    — float32, shape (1, 6)
```

## EfficientNet-B0 Backbone

The backbone uses compound scaling with the following configuration:

| Parameter | Value |
|-----------|-------|
| Width coefficient | 1.0 |
| Depth coefficient | 1.0 |
| Resolution | 224 → adapted to 864×768 |
| Dropout rate | 0.2 |
| Drop connect rate | 0.2 |

## Heatmap Head

```
features (1280 ch) → Conv2d(1280, 256, 3×3, pad=1) → BatchNorm2d → ReLU
                   → Conv2d(256, 29, 1×1) → Sigmoid
```

## CVM Head

```
features (1280 ch) → GlobalAvgPool2d → Dropout(0.2) → Linear(1280, 6)
```

## Inference Providers

ONNX Runtime supports multiple execution providers:

| Provider | Use Case |
|----------|----------|
| `CPUExecutionProvider` | Default, works everywhere |
| `CUDAExecutionProvider` | NVIDIA GPU acceleration |
| `TensorrtExecutionProvider` | NVIDIA TensorRT (fastest) |
| `CoreMLExecutionProvider` | Apple Silicon (M1/M2/M3) |
| `OpenVINOExecutionProvider` | Intel hardware |
| `DirectMLExecutionProvider` | Windows GPU (AMD, Intel, NVIDIA) |
