# Sports Image Classification — Transfer Learning vs. Custom CNN

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

## What the Data Shows

### Model Architectures

**Xception (Transfer Learning)**
- Base: Xception pretrained on ImageNet — all 20M+ weights frozen
- Head: GlobalAveragePooling2D → Dropout(0.25) → Dense(100, softmax)
- Only the 100-unit classification head is trained from scratch
- Converges in 20 epochs on a Tesla T4 GPU

**Custom CNN (From Scratch)**
- 4 convolutional blocks: 64 → 128 → 256 → 512 filters
- Each block: Conv2D × 2 + BatchNormalization + MaxPooling + Dropout
- GlobalAveragePooling2D → Dense(256) → Dropout(0.5) → Dense(100, softmax)
- ~4.85M trainable parameters, trained for 30 epochs

### Training Progression (Xception)

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 54.7% | 71.2% |
| 7 | 96.6% | 90.4% |
| 11 | 98.8% | 94.4% |
| 20 | **100.0%** | **95.2%** |
| **Test** | — | **97.6%** |

### Failure Analysis

The confusion matrix reveals that both models struggle most with visually similar sports that share the same background environment — soccer, rugby, and American football are occasionally confused. Sports with highly distinctive visual signatures (e.g., sumo wrestling, swimming) are classified near-perfectly.

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
├── BI Term Paper.pdf                  # Full written report
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

## Context

Uzun, B., Yalcin, M., Akalin, A. (2024). *Sports Image Classification using Transfer Learning and Custom CNNs.* Business Intelligence II, University of Oldenburg.

Methods: Convolutional Neural Networks, Transfer Learning (Xception/ImageNet), Batch Normalization, Dropout regularization, EarlyStopping, ReduceLROnPlateau.
