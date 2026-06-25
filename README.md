# Knee Osteoarthritis (KOA) Severity Grading with GAN Augmentation & Deep Learning

An end-to-end pipeline for automatic **Kellgren–Lawrence (KL)** grading of knee
osteoarthritis from plain X-rays. The project combines classical image
enhancement, **GAN-based data augmentation targeted at the under-represented
minority grades (KL 1 and 2)**, an ensemble of nine CNN/Transformer backbones
with attention fusion and stacking, Grad-CAM explainability, and a clinical
decision-support GUI that produces a printable patient report.

---

## Table of Contents

- [Overview](#overview)
- [Development Timeline (9 Steps)](#development-timeline-9-steps)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Module Details](#module-details)
  - [1. Preprocessing](#1-preprocessing)
  - [2. DCGAN — Synthetic X-ray Generation (Grades 1 & 2)](#2-dcgan--synthetic-x-ray-generation-grades-1--2)
  - [3. SRGAN — Super-Resolution](#3-srgan--super-resolution)
  - [4. Models — Classification](#4-models--classification)
  - [5. GUI — Inference & Clinical Decision Support](#5-gui--inference--clinical-decision-support)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Notes & Limitations](#notes--limitations)
- [Disclaimer](#disclaimer)

---

## Overview

Knee osteoarthritis is graded on the 5-point Kellgren–Lawrence scale:

| KL Grade | Meaning |
|----------|---------|
| 0 | Normal — no radiographic OA |
| 1 | Doubtful — minimal changes |
| 2 | Mild — definite osteophytes |
| 3 | Moderate — joint space narrowing |
| 4 | Severe — large osteophytes, marked narrowing |

The dataset is **imbalanced**, with grades **1 and 2 the most under-represented**,
and some source images are low-resolution. To address this the project:

1. Trains **per-grade DCGANs for the minority grades 1 and 2 only** to synthesise
   additional X-rays.
2. **Super-resolves** those 64×64 synthetic images up to 224×224 with an SRGAN.
3. Enhances images with a fixed **classical filter chain** (γ = 1.2).
4. Trains a diverse **model zoo** that is combined via **multi-head attention
   fusion** and a **stacking ensemble**, with **Grad-CAM** explainability.

Two label schemes are supported:

- **5-class** — full KL 0–4.
- **4-class** — grades 0 and 1 merged (the clinically ambiguous "normal vs.
  doubtful" distinction), which yields the best accuracy.

Adding the GAN-generated minority-grade images grows the training set from
**5,778 → 7,788** images (+2,010).

---

## Development Timeline (9 Steps)

The project was built in nine chronological steps across three phases. This is
the order to follow when reading the notebooks/scripts:

```
V1 → V3 → V8 → DCGAN → SRGAN → 5-class → 4-class → 5-class ES → 4-class ES
```

| Step | Experiment | Phase | GAN in training? | Best result |
|------|------------|-------|------------------|-------------|
| 1 | Version 1 | Pre-GAN (filtered, 5,778) | No | VGG16 68.90% |
| 2 | Version 3 | Pre-GAN baseline | No | VGG16 69.44% |
| 3 | Version 8 | Pre-GAN + top-3 fusion | No | Fusion (MHA) 70.59% |
| 4 | **DCGAN** | GAN data pipeline | Generates grade 1 & 2 data | — |
| 5 | **SRGAN** | GAN data pipeline | Upscales GAN data | Val PSNR 38.90 dB |
| 6 | 5-class (no early stop) | With GANs (7,788) | Yes | Hard Voting 72.40% |
| 7 | 4-class (no early stop) | With GANs (7,788) | Yes | Hard Voting 82.67% |
| 8 | 5-class + early stop | With GANs (7,788) | Yes | Hard Voting 72.10% |
| 9 | 4-class + early stop ⭐ | With GANs (7,788) | Yes | **Stacking (LR) 83.15%** |

- **Phase 1 (steps 1–3):** progressive pre-GAN CNN baselines on filtered data,
  ending with a top-3 multi-head attention fusion (Version 8).
- **Phase 2 (steps 4–5):** the GAN data pipeline — DCGAN generation for grades
  1 & 2, then SRGAN super-resolution.
- **Phase 3 (steps 6–9):** the final classifiers trained on the GAN-augmented
  dataset, in 5-class / 4-class × with / without early-stopping variants.

---

## Pipeline

```
Raw X-rays
   │
   ▼
[1] Preprocessing  ──►  contrast stretch → histogram equalisation → gamma (γ=1.2) → sharpen
   │
   ▼
[2] DCGAN          ──►  per-grade synthetic 64×64 X-rays for MINORITY grades 1 & 2
   │
   ▼
[3] SRGAN          ──►  upscale 64×64 → 256×256 → 224×224 (×4 super-resolution)
   │
   ▼
[4] Models         ──►  9 backbones + MHA fusion + stacking ensemble + Grad-CAM
   │
   ▼
[5] GUI            ──►  Streamlit app: prediction, Grad-CAM, CDSS, PDF report
```

---

## Project Structure

```
.
├── Preprocessing/
│   └── main Filters.py            # Batch image-enhancement filters (γ default 1.2)
├── DCGANS/
│   └── local_dcgan_m2_top5.py     # DCGAN training/generation (Apple M-series MPS)
├── SRGANS/
│   └── srgans.ipynb               # SRGAN super-resolution (Kaggle P100)
├── Models/
│   ├── Version 1.ipynb            # Step 1 · Pre-GAN baseline (6 CNNs, filtered)
│   ├── version 3 best acc.ipynb   # Step 2 · Pre-GAN baseline (6 CNNs)
│   ├── version-8.ipynb            # Step 3 · Pre-GAN + top-3 MHA fusion
│   ├── 5 Classes.ipynb            # Step 6 · 5-class, GAN, no early stop (6 backbones)
│   ├── 4Classes.ipynb             # Step 7 · 4-class, GAN, no early stop (6 backbones)
│   ├── 5 Classes early stop.ipynb # Step 8 · 5-class, early stop (9 backbones)
│   └── 4 classes early stop.ipynb # Step 9 · 4-class, early stop (9 backbones) ⭐ best
└── GUI/
    ├── app_streamlit.py           # Streamlit front end
    └── inference_models.py        # Model definitions, loaders, ensemble, Grad-CAM
```

> **Note:** trained weight files (`*.pth`), generated/super-resolved images, and
> result plots are produced by the notebooks/scripts and are expected to live in
> the matching experiment folders (e.g. `Models/4 Classes early stop/`). They are
> not bundled with the source code due to size. The raw dataset is the public
> [Knee Osteoarthritis Severity Grading Dataset (Mendeley Data)](https://data.mendeley.com/datasets/56rmx5bjcr/1).

---

## Module Details

### 1. Preprocessing

`Preprocessing/main Filters.py` applies a fixed enhancement chain to every image
in a folder and writes the results to an output folder:

1. **Contrast stretching** — rescales intensities to the full 0–255 range.
2. **Histogram equalisation** — improves global contrast (luma channel for colour).
3. **Gamma correction** — **γ = 1.2** (project standard, the function default).
4. **Unsharp masking** — edge sharpening.

The same enhancement chain is reproduced inside the GUI so that inference matches
the training distribution. This filtering is used consistently across every
experiment.

### 2. DCGAN — Synthetic X-ray Generation (Grades 1 & 2)

`DCGANS/local_dcgan_m2_top5.py` trains a **Deep Convolutional GAN per KL grade**
to generate 64×64 grayscale X-rays for augmentation. Although the code supports
grades 0–4, **only grades 1 and 2 were actually trained and used**, because they
are the most under-represented minority classes where real data is scarcest.

- DCGAN generator/discriminator at 64×64, single channel; latent dim 100.
- Trains one grade at a time; auto-detects Apple **MPS**, CUDA, or CPU.
- Up to 650 epochs, batch 64, Adam (lr 2e-4); checkpoints every 25 epochs.
- Keeps a **top-5** record of the best epochs by combined G+D loss
  (`top5_losses_grade{1,2}.json`), with preview grids, then generates any number
  of images from a selected best checkpoint.
- Output images use the dataset naming style (`#######L/R.png`) so they drop
  straight into the training folders.

**Generated counts used downstream:** grade 1 → **1,240** images (epoch 249,
selected for visual quality); grade 2 → **770** images (epoch 332, lowest loss).

### 3. SRGAN — Super-Resolution

`SRGANS/srgans.ipynb` trains a **Super-Resolution GAN** (SRResNet generator +
VGG19 perceptual loss + adversarial loss) that maps the 64×64 DCGAN outputs up to
256×256 (×4) and resizes to 224×224 for the classifiers, restoring fine detail
before classification.

- Built for Kaggle with a Tesla **P100** (the notebook pins a CUDA 11.8 /
  Pascal-compatible PyTorch build).
- 5 epochs SRResNet pretrain + 50 epochs full SRGAN; loss = 1.0·MSE +
  1e-3·adversarial + 6e-3·VGG perceptual.
- **Best validation PSNR 38.90 dB** (epoch 35). It upscales the same
  1,240 grade-1 and 770 grade-2 images produced by the DCGAN.

### 4. Models — Classification

The `Models/` notebooks contain the full classifier timeline. Steps 1–3 are
**pre-GAN baselines** (filtered data, 5,778 train); steps 6–9 are the
**GAN-augmented** classifiers (7,788 train). The shared architecture and loaders
are mirrored in `GUI/inference_models.py`.

**Backbones (up to 9 in the GAN experiments):**

| Backbone | Family |
|----------|--------|
| VGG-16, VGG-19 | CNN |
| DenseNet-121, DenseNet-201 | CNN |
| ConvNeXt-Tiny | CNN |
| EfficientNet-V2-S | CNN |
| RegNetY-008 | CNN |
| Swin Transformer Tiny | Transformer |
| ViT-Base/16 | Transformer |

The pre-GAN baselines (V1/V3/V8) instead use a classic set of 6 torchvision
models (VGG16/19, ResNet101, MobileNetV2, InceptionV3, DenseNet121); Version 8
adds a top-3 attention fusion. Each backbone uses a custom classification head
(1024→512→256→classes with BatchNorm + Dropout).

**Combination strategies (GAN steps 6–9):**

- **Multi-Head Attention Fusion (MHA)** — projects the top-3 backbones' features
  into a shared 512-d embedding, adds a CLS token + positional embeddings, and
  fuses them through a 2-layer, 8-head Transformer encoder.
- **Ensemble (5 methods)** — Hard Voting, Soft Voting, Weighted Soft Voting, Max
  Confidence, and **Stacking (Logistic Regression)** meta-learner over the
  concatenated per-class probabilities of all backbones (+ fusion).

**Training (steps 6–9):** 50 epochs, batch 32, Adam (lr 1e-4, wd 1e-4),
ReduceLROnPlateau; CrossEntropy with label smoothing 0.1 (FocalLoss γ=2 for
ConvNeXt); early-stop variants use patience 10.

**Explainability:** Grad-CAM is supported for every backbone with
architecture-aware target layers (CNN vs. Transformer), plus accuracy-weighted
Grad-CAM for the ensemble and averaged Grad-CAM for the fusion.

### 5. GUI — Inference & Clinical Decision Support

`GUI/app_streamlit.py` is a Streamlit application that ties the GAN classifiers
(steps 6–9) together. It exposes the four trained experiments via
`EXPERIMENT_PROFILES` in `inference_models.py`:

- Select training experiment + classifier (ensemble, fusion, or any single backbone).
- Upload an X-ray; it is preprocessed with the same filter chain (γ default 1.2,
  adjustable) and classified.
- Shows predicted KL grade, confidence, top-3 and full probability distribution.
- **Grad-CAM heatmap** overlay of where the model focused.
- **Hybrid Clinical Decision Support System (CDSS):** takes patient inputs (age,
  BMI, pain, walking difficulty, comorbidities, activity level), computes
  progression / disability / obesity / surgical-need risk scores, and generates
  personalised, confidence-aware recommendations.
- **PDF export** — a styled clinical report with images, Grad-CAM, risk grid,
  probability table, and recommendations (requires `fpdf2`).

The **default experiment is the 4-class + early-stop stacking ensemble** (best
overall accuracy).

---

## Results

Best reported test accuracy per experiment:

| Step | Experiment | Classes | Best method | Test Acc |
|------|------------|---------|-------------|----------|
| 9 | 4-class + early stop ⭐ | 4 (0+1 merged) | Stacking (LR) | **83.15%** |
| 9 | 4-class + early stop | 4 | Fusion (MHA) | 82.85% |
| 7 | 4-class | 4 | Hard Voting | 82.67% |
| 6 | 5-class | 5 (KL 0–4) | Hard Voting | 72.40% |
| 8 | 5-class + early stop | 5 | Hard Voting | 72.10% |
| 3 | Version 8 (pre-GAN) | 5 | Fusion (MHA) | 70.59% |
| 2 | Version 3 (pre-GAN) | 5 | VGG16 | 69.44% |
| 1 | Version 1 (pre-GAN) | 5 | VGG16 | 68.90% |

Key observations:

- **GAN augmentation + 4-class merging** lifts accuracy from ~69–71% (pre-GAN,
  5-class) to **~83%** (GAN, 4-class).
- The **4-class + early-stop stacking ensemble (83.15%)** is the default and
  recommended configuration in the GUI.
- SRGAN validation PSNR (38.90 dB) is marginally below the bicubic baseline
  (39.24 dB); the adversarial/perceptual training targets perceptual detail
  rather than PSNR alone.

---

## Installation

Requires **Python 3.10+**.

```bash
# (recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# core dependencies
pip install torch torchvision timm streamlit opencv-python pillow numpy matplotlib tqdm scikit-learn seaborn pandas

# optional: PDF report export in the GUI
pip install fpdf2
```

> GPU is optional. PyTorch automatically uses CUDA (NVIDIA) or MPS (Apple
> Silicon) when available, otherwise CPU. Some notebooks pin `numpy<2`; the
> SRGAN notebook installs a CUDA 11.8 / Pascal-compatible PyTorch build for the
> Kaggle P100.

---

## Usage

### Run the GUI

```bash
cd GUI
streamlit run app_streamlit.py
```

Then open the local URL Streamlit prints, pick an experiment + model, upload a
knee X-ray, fill in patient details, and (optionally) export the PDF report.
The app expects trained `.pth` checkpoints in the matching `Models/` experiment
folders.

### Preprocess a folder of images

Edit `INPUT_FOLDER`, `OUTPUT_FOLDER`, and `GAMMA_VALUE` (project standard 1.2) at
the bottom of `Preprocessing/main Filters.py`, then:

```bash
python "Preprocessing/main Filters.py"
```

### Train / generate with DCGAN (grades 1 & 2)

```bash
cd DCGANS
python local_dcgan_m2_top5.py
# 1 = train a chosen grade (use 1 or 2), 2 = generate from the top-5 best models
```

Expects a dataset at `DCGANS/Original Dataset train/<grade>/` (ImageFolder
layout with grade sub-folders).

### Super-resolution & classification

Open `SRGANS/srgans.ipynb` and the `Models/*.ipynb` notebooks in
Jupyter/Kaggle and run the cells top to bottom, following the chronological
order above. The model notebooks were built for a CUDA GPU (Kaggle P100 / T4).

---

## Notes & Limitations

- **GAN augmentation targets grades 1 & 2 only** — the most under-represented
  classes; grades 0, 3, 4 were not GAN-augmented.
- Trained weights and generated/super-resolved images are not included in the
  repository due to size; download the [dataset](https://data.mendeley.com/datasets/56rmx5bjcr/1)
  and run the scripts/notebooks to reproduce them.
- The notebooks/scripts contain hard-coded dataset paths that may need to be
  adapted to your environment.
- No automated tests, and a single train/val/test split (no cross-validation).
- The clinical risk scores and recommendations in the GUI are rule-based and
  intended for decision *support*, not diagnosis.
- 5-class accuracy remains modest (~72%); the 4-class merge is what reaches ~83%.

---

## Disclaimer

This system is an **AI-assisted** research and educational tool. It is **not a
medical device** and must **not** replace professional clinical judgement. All
findings must be validated by a qualified healthcare professional before any
treatment decision.
