# Sports Image Classification — Transfer Learning vs. Custom CNN

<img width="1800" height="379" alt="image" src="https://github.com/user-attachments/assets/39430438-6bc3-4667-a7a8-de7196ff3525" />

**Can a frozen pretrained network outperform a carefully designed CNN built from scratch on a 100-class image classification task?**

This project trains and compares two deep learning architectures on the [Kaggle Sports Classification dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification): an Xception model leveraging ImageNet transfer learning, and a custom 4-block CNN trained from random initialization.

📄 [Read the full paper](./Paper.pdf)

---

## Key Results

| Model | Architecture | Test Accuracy | Precision | Recall |
|-------|-------------|---------------|-----------|--------|
| **Xception** | Pretrained ImageNet + custom head | **97.6%** | **98.83%** | **97.00%** |
| Custom CNN | 4-block Conv (64→512) + BatchNorm + Dropout | 76.4% | — | — |

**Transfer learning delivers a +21.2 pp accuracy gain** over a well-regularized from-scratch CNN — the gap is the point. Both models are trained on the same data; the difference is entirely attributable to pretrained representations.

---

## Background: Why Image Classification Is Hard

A raw image is a 3D array of pixel intensity values — 224 × 224 × 3 = **150,528 numbers** per image. The challenge is that the same object (e.g. a tennis player) can appear at different positions, scales, lighting conditions, and angles. A model that memorizes pixel positions will fail immediately on new images.

The key insight of **Convolutional Neural Networks (CNNs)** is to exploit the spatial structure of images: nearby pixels are related, and the same feature (an edge, a curve, a texture) can appear anywhere in the image. CNNs learn *where* to look for these features automatically, rather than being told.

---

## Convolutional Neural Networks

### How Convolution Works

A convolutional layer applies a set of small learnable **filters** (kernels) across the entire image. Each filter slides over the input and computes a dot product at every position, producing a **feature map** that highlights where that pattern appears:

```math
(I * K)(i,j) = \sum_{m}\sum_{n} I(i+m,\, j+n) \cdot K(m,n)
```

where $I$ is the input image, $K$ is the filter kernel, and $(i,j)$ is the spatial position. A filter that detects vertical edges will produce high activation values wherever a vertical edge exists in the image.

### Feature Hierarchy

Stacking multiple convolutional layers creates a **feature hierarchy**:

| Layer depth | What it detects |
|-------------|----------------|
| Early layers | Edges, corners, color gradients |
| Middle layers | Textures, shapes, object parts |
| Deep layers | High-level concepts: faces, wheels, court markings |

This hierarchy is why deep networks work: the network progressively builds abstract representations from raw pixels.

### Pooling

After each convolutional block, **MaxPooling** downsamples the feature maps by taking the maximum value in each local region. This reduces spatial dimensions (saving computation) while retaining the most activated features and providing spatial invariance — the network becomes less sensitive to exact object position.

### GlobalAveragePooling vs. Flatten

Before the final classification layer, the spatial feature maps must be converted to a 1D vector. **Flatten** concatenates all values (large vector, prone to overfitting). **GlobalAveragePooling2D** averages each feature map into a single number — a much more compact representation that acts as a form of regularization and reduces the number of parameters significantly.

---

## Transfer Learning

### The Core Idea

Training a deep CNN from scratch requires millions of labeled images and days of GPU compute. **Transfer learning** bypasses this by reusing a network already trained on a large dataset (ImageNet — 1.2 million images, 1,000 classes).

The pretrained network has already learned a rich visual vocabulary: edges, textures, shapes, object parts. These representations transfer well to new tasks because visual features are largely universal. The pretrained weights are **frozen** — only a small new classification head is trained on the target dataset.

```
ImageNet pretrained base (frozen weights)
        ↓
  [Rich visual features: edges → textures → shapes → parts]
        ↓
  GlobalAveragePooling2D
        ↓
  Dropout(0.25)
        ↓
  Dense(100, softmax)   ← only this is trained from scratch
```

### Why It Works on 14,493 Images

Training a 20M+ parameter network from scratch on ~14,000 images would severely overfit. Transfer learning sidesteps this: the base extracts features for free, and only ~100K parameters in the head need to be learned — a much more tractable problem for a small dataset.

---

## The Xception Architecture

Xception (*Extreme Inception*) is a CNN introduced by François Chollet (Google, 2017) that replaces standard convolutions with **depthwise separable convolutions**:

- A standard convolution mixes spatial filtering and channel combination in one step
- Depthwise separable convolutions split these into two sequential operations: (1) spatial filtering per channel, (2) pointwise 1×1 convolution to combine channels

This factorization reduces parameters and computation while maintaining or improving accuracy. Xception achieves higher ImageNet accuracy than the original Inception V3 with a similar parameter count (~22M parameters).

---

## Regularization Techniques

### Batch Normalization

After each convolutional or dense layer, **Batch Normalization** normalizes the layer's activations across the mini-batch:

```math
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
```

where $\mu_B$ and $\sigma_B^2$ are the batch mean and variance. The network then applies learned scale ($\gamma$) and shift ($\beta$) parameters. This stabilizes and accelerates training by reducing **internal covariate shift** — the problem that the distribution of each layer's inputs changes as earlier layers update their weights.

### Dropout

**Dropout** randomly sets a fraction $p$ of neuron activations to zero during each training step. This forces the network to learn redundant representations and prevents units from co-adapting:

```math
\tilde{h}_i = h_i \cdot \text{Bernoulli}(1 - p)
```

At test time, all neurons are active and outputs are scaled by $(1-p)$. The custom CNN uses Dropout(0.5) in the dense layers; the Xception head uses Dropout(0.25). Without dropout, the custom CNN overfit heavily on this small dataset.

### EarlyStopping & ReduceLROnPlateau

- **EarlyStopping**: halts training when validation loss stops improving for a set number of epochs (`patience`), preventing overfitting past the optimal point
- **ReduceLROnPlateau**: reduces the learning rate by a factor when validation loss plateaus, allowing finer weight updates when approaching convergence

---

## Evaluation Metrics

### Accuracy

The fraction of correctly classified images across all 100 classes:

```math
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}}
```

Accuracy is informative here because the test set is relatively balanced across classes. In heavily imbalanced settings it can be misleading (see: the PD credit risk project).

### Precision and Recall (per class, macro-averaged)

For each class $c$:

```math
\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \qquad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}
```

- **Precision**: of all images predicted as class $c$, how many actually are?
- **Recall**: of all actual class $c$ images, how many were correctly identified?

Macro-averaged precision and recall average these values across all 100 classes equally, giving each sport equal weight regardless of how many images it has.

### Categorical Cross-Entropy Loss

Both models are trained to minimize **categorical cross-entropy**, which measures the divergence between predicted probability distributions and true one-hot labels:

```math
\mathcal{L} = -\sum_{c=1}^{100} y_c \cdot \log(\hat{p}_c)
```

where $y_c \in \{0,1\}$ is the true label for class $c$ and $\hat{p}_c$ is the predicted probability. The **softmax** output layer ensures the 100 class probabilities sum to 1.

---

## Model Architectures

### Xception (Transfer Learning)

- **Base:** Xception pretrained on ImageNet — all 20M+ weights frozen
- **Head:** GlobalAveragePooling2D → Dropout(0.25) → Dense(100, softmax)
- **Trainable parameters:** ~100K (head only)
- **Converges in:** 20 epochs on Tesla T4 GPU

### Custom CNN (From Scratch)

- **4 convolutional blocks:** 64 → 128 → 256 → 512 filters
- **Each block:** Conv2D × 2 + BatchNormalization + MaxPooling + Dropout
- **Head:** GlobalAveragePooling2D → Dense(256) → Dropout(0.5) → Dense(100, softmax)
- **Total trainable parameters:** ~4.85M
- **Trained for:** 30 epochs

---

## Training Progression (Xception)

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 54.7% | 71.2% |
| 7 | 96.6% | 90.4% |
| 11 | 98.8% | 94.4% |
| 20 | **100.0%** | **95.2%** |
| **Test** | — | **97.6%** |

The gap between val accuracy (95.2%) and test accuracy (97.6%) is narrow, confirming the model generalizes well rather than overfitting to the validation set.

---

## Failure Analysis

The confusion matrix reveals that both models struggle most with **visually similar sports sharing the same background environment** — soccer, rugby, and American football are occasionally confused because all three feature a grass field with players in motion. Sports with highly distinctive visual signatures (sumo wrestling, swimming, rock climbing) are classified near-perfectly.

This is consistent with the feature hierarchy view: the network correctly identifies the *environment* but occasionally fails to distinguish the *sport* within that environment — a subtler semantic distinction that would require finer-grained features or more training data in the confused classes.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — Sports Classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification) |
| Total images | 14,493 |
| Classes | 100 sport categories |
| Train / Val / Test | ~70% / 15% / 15% |
| Image size | 224 × 224 × 3 (RGB) |
| Class balance | Moderately imbalanced — soccer/basketball overrepresented |

---

## Repository Structure

```
├── sport_image_classification.ipynb   # Full analysis notebook (6 sections)
├── Paper.pdf                          # Full written report
├── BI_2_presentation.pdf              # Presentation slides
└── README.md
```

---

## How to Run

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn scikit-image

jupyter notebook sport_image_classification.ipynb
```

> **Colab note:** The notebook was developed on Google Colab with a Tesla T4 GPU. Paths reference Google Drive (`/content/drive/MyDrive/`). To run locally, download the dataset from Kaggle and update `train_dir`, `test_dir`, and `valid_dir` in Section 2.

---

Uzun, B., Yalcin, M., Akalin, A. (2024). *Sports Image Classification using Transfer Learning and Custom CNNs.* Business Intelligence II, University of Oldenburg.
