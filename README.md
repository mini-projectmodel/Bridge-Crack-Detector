# Bridge-Crack-Detector
A Deep learning model for Cracked bridges detection 
# 🌉 Bridge Crack Detector

An automated computer vision and deep learning solution designed for structural health monitoring (SHM). This repository provides a trained model capable of detecting, localizing, and classifying surface cracks on concrete bridge structures to assist in predictive infrastructure maintenance and safety inspections.

---

## 📌 Overview

Concrete bridge inspection traditionally relies on manual, labor-intensive visual assessments. **Bridge Crack Detector** automates this process by applying deep learning object detection/segmentation algorithms to high-resolution surface images captured by mobile cameras, robotic crawlers, or drones (UAVs).

### Key Features
- **High-Precision Crack Detection:** Accurately identifies narrow surface cracks, hairline fractures, and surface spalling under varying lighting conditions.
- **Real-Time Inference:** Optimized for rapid processing on edge devices, drones, and local workstations.
- **Flexible Inputs:** Supports batch processing on static images, video feeds, and live camera streams.
- **Export Ready:** Weights convertible to **ONNX**, **TensorRT**, or **TorchScript** for embedded deployment.

---

## 📁 Repository Structure

```text
Bridge-Crack-Detector/
├── data/
│   ├── train/               # Training images & annotations
│   ├── val/                 # Validation dataset
│   └── test/                # Test benchmark samples
├── models/
│   └── best.pt              # Trained weights (PyTorch / YOLO)
├── notebooks/
│   └── training_eda.ipynb   # Exploratory data analysis & training logs
├── src/
│   ├── dataset.py           # Preprocessing & image augmentation pipeline
│   ├── train.py             # Model training script
│   ├── predict.py           # Single/batch inference pipeline
│   └── utils.py             # Metric evaluation & visual plotting tools
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt         # Project dependencies
