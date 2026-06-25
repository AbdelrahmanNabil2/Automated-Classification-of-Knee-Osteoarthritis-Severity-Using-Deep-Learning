# SRGAN — Super-Resolution

Super-Resolution GAN (SRResNet generator + VGG19 perceptual loss + adversarial
loss) that upscales the 64×64 DCGAN outputs to 256×256 (×4) and resizes to
224×224 for the classifiers, restoring fine detail before classification.

## Contents

| File | Description |
|------|-------------|
| `srgans.ipynb` | SRGAN training + super-resolution (built for Kaggle Tesla P100) |

## Highlights

- 5 epochs SRResNet pretrain + 50 epochs full SRGAN.
- Loss = 1.0·MSE + 1e-3·adversarial + 6e-3·VGG perceptual.
- **Best validation PSNR 38.90 dB** (epoch 35).
- Upscales the same 1,240 grade-1 and 770 grade-2 images produced by the DCGAN.

## Usage

Open `srgans.ipynb` in Jupyter/Kaggle and run the cells top to bottom. The
notebook pins a CUDA 11.8 / Pascal-compatible PyTorch build for the P100.

See the [root README](../README.md) for the full pipeline.
