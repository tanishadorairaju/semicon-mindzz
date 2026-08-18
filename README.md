<div align="center">
  <h1>AI-Based Restoration of Degraded Semiconductor Images</h1>
  <p><i>A sleek and powerful AI pipeline to rescue degraded semiconductor scans!</i></p>
</div>

---

## Overview

This project addresses the problem of **AI-based restoration of degraded semiconductor inspection images**. 

Semiconductor inspection images can suffer from multiple forms of degradation, including:
- **Speckle noise**
- **Gaussian noise**
- **Spatial resolution reduction**
- **Loss of fine structural details**

**The Objective:** Develop an AI system that takes a degraded, low-resolution grayscale image as input and produces a clean, high-resolution restored image.

The proposed system uses **NAFNet** as the primary restoration backbone, followed by a **2× PixelShuffle** super-resolution module. A **SwinIR-based alternative** is considered when NAFNet does not provide sufficient restoration quality.

---

## Problem Statement

The training dataset consists of paired images. For every degraded image, a corresponding clean ground-truth image is provided.

```text
Degraded Image                       Ground Truth

 128 × 128                           256 × 256
   Noisy                               Clean
Low Resolution                    High Resolution
       │                                 │
       └──────────── Paired ─────────────┘
```

---

## Degradations Addressed

### 1. Speckle Noise
Speckle noise introduces random intensity variations and grain-like patterns into the image.
* **Model Objective:** Remove the noise while preserving the actual semiconductor structures and fine details. The model should avoid excessive smoothing because blurring can remove important inspection information.

### 2. Gaussian Noise
Gaussian-like degradation can reduce image sharpness and make edges and fine structures less distinct.
* **Model Objective:** Restore sharp edges and structural information without introducing artificial patterns or ringing artifacts.

### 3. Spatial Resolution Reduction
The degraded images have lower spatial resolution than their corresponding ground-truth images. 
For the current dataset:
```text
128 × 128  ->  256 × 256
```
* **Model Objective:** Recover the original image resolution while reconstructing useful fine details lost during downsampling.

---

## Proposed Solution

The proposed pipeline flows as follows:

```text
  Degraded Low-Resolution Image
                │
                ▼
        NAFNet Restoration
                │
                ▼
         2× PixelShuffle
                │
                ▼
  Restored High-Resolution Image
```
*NAFNet performs the primary image restoration, while PixelShuffle performs the 2× spatial upscaling.*

---

## Model Architecture

### Primary Model — NAFNet
NAFNet is used as the primary restoration backbone because it provides a lightweight image restoration architecture suitable for efficient inference. The model is designed to learn restoration directly from paired degraded and ground-truth images.

**Main Components:**
- NAFNet restoration backbone
- Encoder-decoder architecture
- Simplified attention mechanism
- Residual learning
- 2× PixelShuffle super-resolution
- Single-channel grayscale processing

### Alternative Model — SwinIR
During model development, validation performance is monitored to determine whether NAFNet provides sufficient restoration quality. If NAFNet produces insufficient results (particularly on challenging or out-of-distribution images), **SwinIR** can be selected as a higher-capacity alternative.

**Model Selection Strategy:**
```text
                   Degraded Image
                         │
                         ▼
                       NAFNet
                   (PRIMARY MODEL)
                         │
                         ▼
               Validation Performance
                  /                            Good             Poor
                │                 │
                ▼                 ▼
         Continue NAFNet       Evaluate
                                SwinIR
                             (ALTERNATIVE)
```
*This strategy prioritizes NAFNet for efficient deployment while retaining SwinIR for when additional restoration quality is required. Current reported results are based on NAFNet.*

### Super-Resolution
The input image is `128×128` while the target ground-truth image is `256×256`. A 2× PixelShuffle module is used after the restoration backbone.

```text
           NAFNet Output
                 │
                 ▼
            Convolution
          (1 -> 4 channels)
                 │
                 ▼
          PixelShuffle ×2
                 │
                 ▼
         256 × 256 Output
```
*PixelShuffle rearranges feature channels into spatial information, allowing efficient 2× upscaling.*

---
## Training Strategy

The model is trained using paired degraded and ground-truth images.

### Dataset Split
- **Total paired images:** 3,200
- **Training:** 2,880 images
- **Validation:** 320 images
*(The split uses a fixed random seed `random_state = 42` to ensure reproducibility).*

### Training Configuration

| Parameter | Value |
| :--- | :--- |
| **Framework** | PyTorch |
| **Primary Model** | NAFNet |
| **Alternative Model** | SwinIR |
| **Input** | `128×128` grayscale |
| **Output** | `256×256` grayscale |
| **Batch Size** | 16 |
| **Optimizer** | Adam |
| **Learning Rate** | `0.0001` |
| **Loss Function** | L1 Loss |
| **Training Epochs** | 30 |
| **GPU** | NVIDIA Tesla T4 |
| **Acceleration** | CUDA |

### Loss Function
The current implementation uses **L1 reconstruction loss** *(Mean Absolute Error)*. L1 loss encourages the restored output to remain close to the ground-truth image while being relatively robust to large pixel errors.
> *Future improvements can incorporate structural losses such as SSIM-based loss or Charbonnier loss to further preserve fine semiconductor structures.*

### Data Handling
The training dataset contains `.npy` grayscale image arrays. The degraded input can contain intensity values outside the normal `[0,1]` range due to the degradation process. 

During preprocessing, values are clipped to the valid image range to provide a stable numerical range for training:
```python
noisy = np.clip(noisy, 0.0, 1.0)
gt = np.clip(gt, 0.0, 1.0)
```

---

## Evaluation Metrics

The model is evaluated using multiple metrics:

- **PSNR (Peak Signal-to-Noise Ratio):** Measures pixel-level reconstruction quality. *(Higher is better)*
- **SSIM (Structural Similarity Index):** Measures similarity in image structure and visual characteristics. *(Higher is better)*
- **LPIPS (Learned Perceptual Image Patch Similarity):** Measures perceptual differences. *(Lower is better)*
- **Inference Speed:** Measured to determine suitability for high-throughput environments (`ms/image` and `images/sec`).

### Current Results
The current NAFNet validation results are:

| Metric | Result |
| :--- | :--- |
| **Baseline PSNR** | 25.69 dB |
| **NAFNet PSNR** | 22.99 dB |
| **PSNR Difference** | -2.70 dB |
| **SSIM** | 0.6112 |
| **LPIPS** | 0.4778 |
| **Inference Time** | 9.09 ms/image |
| **Throughput** | 110.03 images/sec |

> **Note:** The current lightweight baseline achieves a higher PSNR than the NAFNet implementation. Therefore, further model improvement and tuning are being investigated rather than claiming that NAFNet currently outperforms the baseline.

---

## Generalization & Efficiency

### Generalization
The challenge includes images from different sources and semiconductor structures. Potential strategies for generalization include:
- Degradation-aware augmentation
- Noise-level & Intensity variation
- Random spatial transformations
- Validation on unseen samples
- Model selection based on validation performance

### Efficiency
Inference speed is an important requirement. The lightweight NAFNet architecture is prioritized as the primary model, achieving an **average inference time of 9.09 ms/image** (110.03 images/second) using GPU acceleration through CUDA.

---

## Hardware and Software

**Hardware:**
- **GPU:** NVIDIA Tesla T4 (Training) / potentially NVIDIA H100 (Final Benchmarking)
- **Platform:** Google Colab

**Software Stack:**
`Python` | `PyTorch` | `CUDA` | `NumPy` | `scikit-learn` | `scikit-image` | `LPIPS` | `Pandas` | `Matplotlib` | `OpenCV` | `Pillow`

---

## Repository Structure

```text
semicon-mindzz/
│
├── README.md               # You are here!
├── requirements.txt
├── .gitignore
│
├── training/
│   └── train.py            # Training pipeline
│
├── inference/
│   └── evaluate.py         # Standalone evaluation script
│
├── model/
│   └── nafnet_final.pth    # Trained model weights
│
└── outputs/
    └── restored_test_outputs/
```

---

## Getting Started

### Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### Inference
The standalone evaluation script handles loading the model, processing images (PNG, JPG, BMP, TIFF), and running super-resolution.

```bash
python inference/evaluate.py     --input_dir ./test_images     --output_dir ./restored
```

### Training
The training pipeline (`training/train.py`) will reproduce the entire workflow from Dataset Loading -> Paired Image Matching -> DataLoader -> NAFNet -> PixelShuffle -> Loss & Optimizer -> Validation & Checkpoint saving. 
*(Note: Training dataset is omitted from the repo due to size restrictions).*

---

## Team & Hackathon Submission

This project was developed for the **AI-Based Restoration of Degraded Images** problem statement. The system focuses on simultaneous noise removal, super-resolution, structure preservation, and efficient inference.

* **Team:** Mindzz

### References
- **NAFNet:** Nonlinear Activation Free Network for Image Restoration
- **SwinIR:** Image Restoration Using Swin Transformer
- PyTorch, LPIPS, scikit-image, CUDA
