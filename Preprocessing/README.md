# Preprocessing

Classical image-enhancement filter chain applied to every X-ray before training
and inference, so the model always sees the same distribution.

## Contents

| File | Description |
|------|-------------|
| `main Filters.py` | Batch enhancement of an input folder → output folder |

## Filter chain

1. **Contrast stretching** — rescale intensities to the full 0–255 range.
2. **Histogram equalisation** — improve global contrast (luma channel for colour).
3. **Gamma correction** — project standard **γ = 1.2** (the function default).
4. **Unsharp masking** — edge sharpening.

## Usage

Edit `INPUT_FOLDER`, `OUTPUT_FOLDER`, and `GAMMA_VALUE` at the bottom of the
script, then run:

```bash
python "Preprocessing/main Filters.py"
```

> The same chain is reproduced inside the GUI (`GUI/app_streamlit.py`) so
> inference matches training. See the [root README](../README.md) for the full
> pipeline.
