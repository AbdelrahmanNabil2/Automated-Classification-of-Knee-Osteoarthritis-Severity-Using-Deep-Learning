# Models — Classification

The full classifier timeline. Steps 1–3 are **pre-GAN baselines** (filtered
data, 5,778 train); steps 6–9 are the **GAN-augmented** classifiers (7,788
train). The shared architecture and loaders are mirrored in
`../GUI/inference_models.py`.

## Notebooks (chronological order)

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `Version 1.ipynb` | Pre-GAN baseline (6 CNNs, filtered) |
| 2 | `version 3 best acc.ipynb` | Pre-GAN baseline (6 CNNs) |
| 3 | `version-8.ipynb` | Pre-GAN + top-3 multi-head attention fusion |
| 6 | `5 Classes.ipynb` | 5-class, GAN, no early stop (6 backbones) |
| 7 | `4Classes.ipynb` | 4-class, GAN, no early stop (6 backbones) |
| 8 | `5 Classes early stop.ipynb` | 5-class, early stop (9 backbones) |
| 9 | `4 classes early stop.ipynb` ⭐ | 4-class, early stop (9 backbones) — best |

## Backbones (up to 9)

VGG-16/19, DenseNet-121/201, ConvNeXt-Tiny, EfficientNet-V2-S, RegNetY-008,
Swin Transformer Tiny, ViT-Base/16.

## Combination strategies

- **Multi-Head Attention Fusion (MHA)** over the top-3 backbones.
- **Ensemble (5 methods):** Hard / Soft / Weighted Soft Voting, Max Confidence,
  and **Stacking (Logistic Regression)** — the best overall (**83.15%**, step 9).

## Notes

- Notebooks were built for a CUDA GPU (Kaggle P100 / T4).
- Trained weights (`*.pth`), `model_comparison_results.csv`, and result plots
  are produced when you run the notebooks and live in matching experiment
  subfolders (e.g. `Models/4 Classes early stop/`). They are **not** committed
  (see `.gitignore`). The GUI expects these files to be present to load models.

See the [root README](../README.md) for full results and architecture details.
