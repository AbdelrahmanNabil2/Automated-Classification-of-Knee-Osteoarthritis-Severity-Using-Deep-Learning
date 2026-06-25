# GUI — Inference & Clinical Decision Support

Streamlit application that ties the GAN classifiers (steps 6–9) together for
prediction, explainability, and a clinical decision-support report.

## Contents

| File | Description |
|------|-------------|
| `app_streamlit.py` | Streamlit front end |
| `inference_models.py` | Model definitions, loaders, ensemble, Grad-CAM |

## Features

- Select training experiment + classifier (ensemble, fusion, or any backbone).
- Upload an X-ray; it is preprocessed with the same filter chain (γ default 1.2).
- Predicted KL grade, confidence, top-3 and full probability distribution.
- **Grad-CAM** heatmap overlay (per-backbone, fused, and accuracy-weighted ensemble).
- **Hybrid CDSS:** rule-based progression / disability / obesity / surgical-need
  risk scores with personalised, confidence-aware recommendations.
- **PDF export** of a styled clinical report (requires `fpdf2`).

The **default experiment is the 4-class + early-stop stacking ensemble** (best
overall accuracy).

## Usage

```bash
cd GUI
streamlit run app_streamlit.py
```

> The app loads trained `.pth` checkpoints from the matching `../Models/`
> experiment folders. Those weights are not committed — train via the
> notebooks first. See the [root README](../README.md).
