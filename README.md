# Masked Autoencoder (MAE) Implementation from Scratch 🧠

> **"Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)**
> *A PyTorch implementation of the Vision Transformer (ViT) architecture that learns visual representations by reconstructing missing pixels.*

---

## 🚀 Project Overview

This project implements a **Masked Autoencoder (MAE)** from scratch. The model learns to understand images without labels by solving a difficult reconstruction task: **hiding 75% of an image** and forcing the neural network to imagine the missing parts based on context.

### Key Features
* **Architecture:** Asymmetric Encoder-Decoder design (Heavy Encoder, Lightweight Decoder).
* **Efficiency:** The Encoder processes *only* the visible patches (25%), reducing computation by 3x-4x compared to standard ViT.
* **Self-Supervised Learning:** Trained entirely on unlabeled data.

---

## 📊 Results & Performance

| Metric | Value | Note |
| :--- | :--- | :--- |
| **Dataset** | **STL-10 (Unlabeled)** | 100k images, 96x96 (upsampled to 224x224) |
| **Epochs** | **100** | Equivalent to ~200 epochs on smaller datasets |
| **Reconstruction Loss** | **MSE** | Minimized mean squared error on invisible patches |
| **Downstream Accuracy** | **60.0%** | Linear Probing on STL-10 Labeled Split |

> **Analysis:** Achieving 60% accuracy with a frozen encoder (Linear Probing) demonstrates that the model successfully learned high-level semantic features (shapes, textures, object parts) purely from the reconstruction task, significantly outperforming a random baseline (10%).

---

## 🖼️ Visualizations

### 1. The Reconstruction Task (25% Visible → Full Image)
The model takes the "Masked Input" (middle) and generates the "Reconstruction" (right).

<img width="100%" alt="MAE Reconstruction Example" src="https://github.com/user-attachments/assets/3be74b56-fbc3-44c9-8937-fc6c3deb3eab">

*(Left: Original Ground Truth | Middle: Masked Input | Right: Model Output)*

### 2. Training Progress
The loss curve shows the model steadily learning to minimize the pixel-wise difference between predicted and actual patches.

<img width="80%" alt="Training Loss Curve" src="https://github.com/user-attachments/assets/9fc6aa9d-6422-4c5e-b31f-0e557cd93cf3">

---

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/AbdelrhmanEldenary/MAE-Project.git](https://github.com/AbdelrhmanEldenary/MAE-Project.git)
cd MAE-Project
