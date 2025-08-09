
# Early Detection of Chronic Kidney Disease (CKD) Using ResNet18 CNN on CT Scan Images

## Project Overview

This project focuses on **early detection and classification of Chronic Kidney Disease (CKD)** using deep learning on CT scan images. We leverage a **ResNet18 convolutional neural network (CNN)** adapted for grayscale medical images to classify kidney CT scans into 4 categories:

- **Cyst**
- **Normal**
- **Stone**
- **Tumor**

The goal is to provide an automated tool to assist medical professionals in diagnosing CKD at an early stage using CT imaging.

---

## Key Features

- Uses **ResNet18 CNN architecture** pretrained on ImageNet, adapted for single-channel grayscale input.
- Classifies CT images into 4 distinct CKD-related categories.
- Applies image preprocessing and normalization suitable for grayscale medical scans.
- Employs data augmentation (random rotation, flips) to improve model generalization.
- Model implemented in PyTorch with support for fine-tuning and early stopping.
- Dataset loaded using `torchvision.datasets.ImageFolder` with class-wise folders.
- Supports training, validation, and inference workflows.

---

## Dataset

- Grayscale CT scan images of kidneys.
- Images organized into four class folders: `Cyst`, `Normal`, `Stone`, `Tumor`.
- Expected folder structure:

  ---
  
## Model Details

Architecture: ResNet18 pretrained on ImageNet.
Input layer: Modified first convolutional layer to accept 1-channel grayscale images.
Output layer: Adjusted for 4-class classification.
Normalization: Mean and std normalization applied (default [0.5], [0.5] or dataset-specific).
Loss function: CrossEntropyLoss.
Optimizer: Adam.

---
## Data Augmentation

Typical augmentations applied:
Random rotations (±10 degrees)
Random horizontal flips

----

## Results

Achieved promising accuracy on validation set (insert your metrics here).
Effective differentiation between cysts, stones, tumors, and normal kidney images.

---

## Future Work

Integrate clinical tabular data with imaging for multimodal learning.
Use dataset-specific normalization for improved performance.
Explore deeper architectures and hyperparameter tuning.
Deploy as a web app using Streamlit or similar frameworks.

---

## Dependencies

Python 3.7+
PyTorch
torchvision
numpy
matplotlib
Pillow (PIL)

---

## Project Structure
.
├── dataset/               # CT scan images (class-wise folders)
├── models/                # Model definitions and saved weights
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── predict.py             # Inference script
├── utils.py               # Helper functions (transforms, loaders)
├── requirements.txt       # Python dependencies
└── README.md              # This file


