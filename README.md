<div align="center">
  <h1>AI-Based Restoration of Degraded Semiconductor Images</h1>
  <p><i>A lightweight AI pipeline for restoring degraded semiconductor inspection images through denoising and super-resolution.</i></p>
</div>

---

## Overview

This project addresses the problem of **AI-based restoration of degraded semiconductor inspection images**.

Semiconductor inspection images can suffer from multiple forms of degradation, including:

* **Speckle noise**
* **Gaussian noise**
* **Spatial resolution reduction**
* **Loss of fine structural details**
* **Reduced edge and feature clarity**

These degradations can make it difficult to accurately inspect semiconductor structures and defects.

### Objective

The objective is to develop an AI system that takes a **degraded, low-resolution grayscale image** as input and produces a **clean, high-resolution restored image**.

The proposed system uses **NAFNet** as the primary restoration backbone, followed by a **2× PixelShuffle** super-resolution module.

A **SwinIR-based alternative** is retained as a higher-capacity option if NAFNet does not provide sufficient restoration quality on challenging or unseen images.

---

## Problem Statement

The training dataset consists of **paired degraded and ground-truth images**.

For every degraded input image, a corresponding clean high-resolution ground-truth image is available.

```text
Degraded Image                       Ground Truth

 128 × 128                           256 × 256
   Noisy                               Clean
Low Resolution                    High Resolution
       │                                 │
       └──────────── Paired ─────────────┘
```

The model therefore learns a direct mapping:

```text
Degraded Low-Resolution Image
              │
              ▼
        AI Restoration
              │
              ▼
Clean High-Resolution Image
```

---

## Degradations Addressed

### 1. Speckle Noise

Speckle noise introduces random intensity variations and grain-like patterns into the image.

**Model Objective:**

Remove the noise while preserving actual semiconductor structures, boundaries, and fine details.

The model should avoid excessive smoothing because aggressive denoising can remove important inspection information.

### 2. Gaussian Noise

Gaussian-like degradation can reduce image sharpness and make edges and fine structures less distinct.

**Model Objective:**

Restore sharp edges and structural information without introducing artificial patterns, ringing artifacts, or excessive smoothing.

### 3. Spatial Resolution Reduction

The degraded images have lower spatial resolution than their corresponding ground-truth images.

For the current dataset:

```text
128 × 128  →  256 × 256
```

**Model Objective:**

Recover the original image resolution while reconstructing useful fine details that were lost during downsampling.

---

## Proposed Solution

The proposed pipeline combines image restoration and super-resolution:

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

**NAFNet** performs the primary image restoration, while **PixelShuffle** performs the 2× spatial upscaling.

The overall system is designed to balance:

* Restoration quality
* Structural preservation
* Model size
* Inference speed
* Suitability for high-throughput inspection environments

---

## Model Architecture

### Primary Model — NAFNet

NAFNet is used as the primary restoration backbone because it provides a lightweight image restoration architecture suitable for efficient inference.

The model learns restoration directly from paired degraded and ground-truth images.

**Main Components:**

* NAFNet restoration backbone
* Encoder-decoder architecture
* Simplified attention mechanism
* Residual learning
* 2× PixelShuffle super-resolution
* Single-channel grayscale processing

### Alternative Model — SwinIR

During model development, validation performance is monitored to determine whether NAFNet provides sufficient restoration quality.

If NAFNet produces insufficient results, particularly on challenging or out-of-distribution images, **SwinIR** can be evaluated as a higher-capacity alternative.

```text
                   Degraded Image
                         │
                         ▼
                       NAFNet
                   (PRIMARY MODEL)
                         │
                         ▼
               Validation Performance
                    /           \
                  Good           Poor
                   │               │
                   ▼               ▼
            Continue NAFNet      Evaluate
                                  SwinIR
                               (ALTERNATIVE)
```

This strategy prioritizes NAFNet for efficient deployment while retaining SwinIR as an alternative when additional restoration capacity is required.

**Current reported results are based on NAFNet.**

---

## Super-Resolution

The input image is `128×128`, while the target ground-truth image is `256×256`.

A **2× PixelShuffle** module is used after the restoration backbone.

```text
           NAFNet Output
                 │
                 ▼
            Convolution
          (1 → 4 channels)
                 │
                 ▼
          PixelShuffle ×2
                 │
                 ▼
         256 × 256 Output
```

PixelShuffle rearranges feature channels into spatial information, allowing efficient 2× upscaling.

---

# Training Strategy

The model is trained using paired degraded and ground-truth images.

## Dataset Split

The current dataset contains **3,200 paired images**.

| Dataset        | Number of Images |
| :------------- | ---------------: |
| **Total**      |            3,200 |
| **Training**   |            2,880 |
| **Validation** |              320 |

The dataset split uses a fixed random seed:

```text
random_state = 42
```

This ensures that the same training and validation samples can be reproduced across experiments.

---

## Training Configuration

| Parameter             | Value                      |
| :-------------------- | :------------------------- |
| **Framework**         | PyTorch                    |
| **Primary Model**     | NAFNet                     |
| **Alternative Model** | SwinIR                     |
| **Input**             | `128×128` grayscale        |
| **Output**            | `256×256` grayscale        |
| **Batch Size**        | 16                         |
| **Optimizer**         | Adam                       |
| **Learning Rate**     | `0.0001`                   |
| **Loss Function**     | 80% Charbonnier + 20% SSIM |
| **Training Epochs**   | 30                         |
| **GPU**               | NVIDIA Tesla T4            |
| **Acceleration**      | CUDA                       |

---

## Loss Function

The initial NAFNet implementation used **L1 reconstruction loss**.

To improve structural preservation and perceptual quality, the loss function was subsequently upgraded to a combination of **Charbonnier Loss** and **SSIM Loss**.

### Improved Loss Configuration

```text
Pixel Loss      : Charbonnier Loss
Structural Loss : SSIM Loss
Combination     : 80% Charbonnier + 20% SSIM
Optimizer       : Adam
Learning Rate   : 0.0001
```

The combined loss is defined as:

```text
Total Loss = 0.8 × Charbonnier Loss + 0.2 × SSIM Loss
```

### Why Charbonnier Loss?

Charbonnier loss is a smooth, robust alternative to traditional L1 loss.

It encourages the model to reduce pixel-level reconstruction errors while being less sensitive to individual large errors.

This is useful for restoration tasks where the model needs to preserve important image structures rather than simply produce a heavily smoothed output.

### Why SSIM Loss?

SSIM focuses on **structural similarity** between the restored image and the ground truth.

This is particularly important for semiconductor inspection because fine edges, boundaries, patterns, and structural features can carry important inspection information.

The combined loss therefore attempts to optimize both:

```text
Pixel Accuracy
      +
Structural Similarity
      ↓
Better Restoration Quality
```

---

## Training Progress

The improved loss configuration was trained for **30 epochs**.

Selected training checkpoints are shown below:

```text
Epoch  1 / 30
Train Loss : 0.091997
Val Loss   : 0.098731

Epoch 10 / 30
Train Loss : 0.081471
Val Loss   : 0.094849

Epoch 20 / 30
Train Loss : 0.073742
Val Loss   : 0.094506

Epoch 30 / 30
Train Loss : 0.068899
Val Loss   : 0.095075
```

The training loss decreased from:

```text
0.091997 → 0.068899
```

over the 30 training epochs.

This indicates that the model progressively learned the restoration mapping from degraded inputs to clean high-resolution targets.

The validation loss improved substantially during the early stages of training and then stabilized, suggesting that the model reached a relatively stable validation performance.

---

## Data Handling

The training dataset contains `.npy` grayscale image arrays.

The degraded input can contain intensity values outside the normal `[0,1]` range due to the degradation process.

During preprocessing, values are clipped to the valid image range to provide a stable numerical range for training:

```python
noisy = np.clip(noisy, 0.0, 1.0)
gt = np.clip(gt, 0.0, 1.0)
```

This ensures that the training data remains numerically stable and compatible with the image restoration pipeline.

---

# Evaluation

The trained model is evaluated on the **320-image validation set**.

Multiple complementary metrics are used rather than relying on a single metric.

## Evaluation Metrics

### PSNR — Peak Signal-to-Noise Ratio

Measures pixel-level reconstruction quality.

**Higher is better.**

PSNR is useful for determining how closely the restored image matches the ground-truth image at the pixel level.

### SSIM — Structural Similarity Index

Measures structural similarity between the restored and ground-truth images.

**Higher is better.**

SSIM is particularly relevant for this project because preserving semiconductor structures and edges is important.

### LPIPS — Learned Perceptual Image Patch Similarity

Measures perceptual differences between images using deep feature representations.

**Lower is better.**

LPIPS provides an additional perspective on perceptual image quality that may not be fully captured by PSNR.

### Inference Speed

Inference speed is measured to determine whether the model can be used in high-throughput environments.

Measurements include:

* `ms/image`
* `images/sec`

---

# Current Results

The NAFNet model was evaluated before and after improving the loss function.

The original implementation used L1 loss, while the improved implementation uses:

```text
80% Charbonnier Loss
+
20% SSIM Loss
```

## Improved NAFNet Results

The improved model achieved the following results on the **320-image validation set**:

| Metric    | Previous NAFNet | Improved NAFNet |
| :-------- | --------------: | --------------: |
| **PSNR**  |        22.99 dB |    **23.58 dB** |
| **SSIM**  |          0.6112 |      **0.6984** |
| **LPIPS** |          0.4778 |      **0.3853** |

### Improvement Summary

```text
PSNR
22.99 dB → 23.58 dB
Improvement: +0.59 dB

SSIM
0.6112 → 0.6984
Improvement: +0.0872

LPIPS
0.4778 → 0.3853
Reduction: -0.0925
```

The results indicate that the improved loss function provided a meaningful improvement over the original NAFNet configuration.

In particular, the increase in SSIM suggests improved structural similarity, while the reduction in LPIPS indicates improved perceptual similarity.

---

## Baseline vs Improved NAFNet

A lightweight CNN baseline was previously trained using the same paired dataset.

The baseline achieved:

```text
Baseline PSNR : 25.69 dB
```

The improved NAFNet achieved:

```text
NAFNet PSNR   : 23.58 dB
```

Comparison:

| Metric         | Lightweight Baseline | Improved NAFNet |
| :------------- | -------------------: | --------------: |
| **PSNR**       |         **25.69 dB** |        23.58 dB |
| **Difference** |                    — |        -2.11 dB |

The baseline currently achieves a higher PSNR than the NAFNet implementation.

However, this does **not** mean that the NAFNet experiment was unsuccessful. The improved NAFNet configuration significantly outperforms the earlier NAFNet configuration:

```text
Previous NAFNet : 22.99 dB
Improved NAFNet : 23.58 dB
                  ↑
              +0.59 dB
```

The improved model also shows stronger SSIM and LPIPS results, demonstrating the benefit of incorporating structural and perceptual considerations into the training objective.

> **Important:** The current results are reported transparently. The lightweight baseline remains the stronger model according to PSNR, while NAFNet remains under investigation for further optimization and deployment-oriented improvements.

---

# Image Output Validation

Additional checks were performed to verify the dimensions and numerical ranges of the model outputs.

## Input Image

```text
Shape : (128, 128)
Min   : 0.0
Max   : 1.0
Mean  : 0.15205136
```

## Ground Truth

```text
Shape : (256, 256)
Min   : 0.0
Max   : 1.0
Mean  : 0.1517643
```

## Model Output

```text
Shape : (256, 256)
Min   : -0.001015434
Max   : 0.91559124
Mean  : 0.1504612
```

The model produces the expected **256×256 output resolution**, confirming that the 2× super-resolution stage is functioning correctly.

The output values are also close to the expected `[0,1]` image range. A very small negative value was observed in the raw model output:

```text
Minimum = -0.001015434
```

This is a minor numerical deviation and can be clipped to `[0,1]` during final image output processing.

---

# Generalization

The challenge includes images from different sources and potentially different semiconductor structures.

A restoration model therefore needs to perform reliably beyond the exact training examples.

Potential strategies for improving generalization include:

* Degradation-aware augmentation
* Noise-level variation
* Intensity variation
* Random spatial transformations
* Validation on unseen samples
* Training with multiple degradation conditions
* Model selection based on validation performance
* Testing on out-of-distribution semiconductor structures

The current implementation provides a foundation for further experimentation with these strategies.

---

# Efficiency

Inference speed is an important requirement for semiconductor inspection systems, where large numbers of images may need to be processed.

The project therefore prioritizes lightweight architectures and efficient inference.

The previously benchmarked NAFNet implementation achieved:

```text
Average Inference Time : 9.09 ms/image
Throughput             : 110.03 images/sec
```

These measurements were obtained using GPU acceleration through CUDA.

The lightweight architecture is intended to make the system more practical for high-throughput inspection environments.

Future benchmarking will evaluate the optimized model and compare performance across different GPU configurations.

---

# Hardware and Software

## Hardware

* **GPU:** NVIDIA Tesla T4 for training
* **Potential Final Benchmark:** NVIDIA H100
* **Platform:** Google Colab

## Software Stack

```text
Python
PyTorch
CUDA
NumPy
scikit-learn
scikit-image
LPIPS
Pandas
Matplotlib
OpenCV
Pillow
```

---

# Repository Structure

```text
semicon-mindzz/
│
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
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

# Getting Started

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Inference

The standalone evaluation script handles loading the trained model, processing supported image formats, and generating restored high-resolution outputs.

Supported formats include:

* PNG
* JPG / JPEG
* BMP
* TIFF

Run:

```bash
python inference/evaluate.py \
    --input_dir ./test_images \
    --output_dir ./restored
```

The pipeline performs:

```text
Input Image
     │
     ▼
Preprocessing
     │
     ▼
NAFNet Restoration
     │
     ▼
2× PixelShuffle
     │
     ▼
Output Clipping / Post-processing
     │
     ▼
Restored 256×256 Image
```

---

# Training

The training pipeline in `training/train.py` reproduces the complete workflow:

```text
Dataset Loading
      ↓
Paired Image Matching
      ↓
DataLoader
      ↓
NAFNet
      ↓
2× PixelShuffle
      ↓
Charbonnier + SSIM Loss
      ↓
Adam Optimizer
      ↓
Validation
      ↓
Checkpoint Saving
```

Example training configuration:

```text
Epochs      : 30
Batch Size  : 16
Optimizer   : Adam
LR          : 0.0001
Loss        : 0.8 Charbonnier + 0.2 SSIM
GPU         : NVIDIA Tesla T4
```

The original training dataset is omitted from the repository due to size restrictions.

---

# Future Improvements

Several improvements can be explored to further increase restoration quality and deployment efficiency.

### Model Improvements

* Further fine-tuning of NAFNet
* Hyperparameter optimization
* Evaluation of SwinIR as a higher-capacity alternative
* Experimentation with additional restoration architectures
* Improved super-resolution reconstruction

### Loss Improvements

* Fine-tuning the Charbonnier/SSIM weighting
* LPIPS-based perceptual loss
* Multi-scale structural losses
* Feature-level reconstruction losses

### Generalization Improvements

* Stronger degradation-aware augmentation
* Multiple noise-level simulation
* Cross-source validation
* Out-of-distribution testing
* Training with additional semiconductor structures

### Deployment Improvements

* FP16 inference
* BF16 inference where supported
* Batch inference
* CUDA optimization
* Model quantization
* GPU benchmarking across Tesla T4 and H100
* Optimization for high-throughput manufacturing environments

---

# Conclusion

This project presents a lightweight AI-based restoration pipeline for degraded semiconductor inspection images.

The system combines:

```text
NAFNet
   +
2× PixelShuffle
   +
Charbonnier Loss
   +
SSIM Structural Loss
   ↓
High-Resolution Image Restoration
```

The improved training configuration achieved:

```text
PSNR : 23.58 dB
SSIM : 0.6984
LPIPS: 0.3853
```

on the 320-image validation set.

Compared with the previous NAFNet configuration, the improved loss function produced:

```text
+0.59 dB PSNR
+0.0872 SSIM
-0.0925 LPIPS
```

The lightweight baseline still achieves a higher PSNR of **25.69 dB**, so further optimization is required before claiming that NAFNet is the best-performing model.

The current implementation therefore focuses on establishing a practical restoration pipeline while providing a foundation for further improvements in **image quality, structural preservation, generalization, and inference efficiency**.

---

# Team & Hackathon Submission

This project was developed for the **AI-Based Restoration of Degraded Images** problem statement.

The system focuses on:

* Noise removal
* Super-resolution
* Fine-structure preservation
* Structural similarity
* Perceptual quality
* Efficient inference
* Practical deployment considerations

### Team

**Mindzz**

---

# References

* **NAFNet:** Nonlinear Activation Free Network for Image Restoration
* **SwinIR:** Image Restoration Using Swin Transformer
* **PyTorch**
* **LPIPS**
* **scikit-image**
* **CUDA**
