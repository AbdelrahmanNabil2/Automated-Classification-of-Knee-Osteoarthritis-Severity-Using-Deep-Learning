# DCGAN — Synthetic X-ray Generation (Grades 1 & 2)

Deep Convolutional GAN trained **per KL grade** to synthesise 64×64 grayscale
knee X-rays for augmenting the under-represented minority grades. Although the
code supports grades 0–4, only **grades 1 and 2** were trained and used.

## Contents

| File | Description |
|------|-------------|
| `local_dcgan_m2_top5.py` | DCGAN training + generation (Apple M-series MPS / CUDA / CPU) |

## Highlights

- Generator/discriminator at 64×64, single channel; latent dim 100.
- Up to 650 epochs, batch 64, Adam (lr 2e-4); checkpoints every 25 epochs.
- Keeps a **top-5** record of the best epochs by combined G+D loss
  (`top5_losses_grade{1,2}.json`) with preview grids.
- Output filenames follow the dataset style (`#######L/R.png`) so they drop
  straight into the training folders.

**Generated counts used downstream:** grade 1 → **1,240** (epoch 249);
grade 2 → **770** (epoch 332).

## Usage

```bash
cd DCGANS
python local_dcgan_m2_top5.py
# 1 = train a chosen grade (1 or 2), 2 = generate from the top-5 best models
```

Expects a dataset at `DCGANS/Original Dataset train/<grade>/` (ImageFolder
layout). Datasets and generated images are **not** committed — see the
[root README](../README.md).
