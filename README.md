# OceanTrace

> **AI-Powered Satellite Oil Spill Detection, Verification & Tracking System**  
> *Internal Hackathon Prototype*

---

## Problem

Marine oil spills are a serious environmental threat. Current detection methods rely on manual inspection of satellite imagery, which is slow, labor-intensive, and delays emergency response. Sentinel-1 SAR satellite data provides consistent global ocean coverage but requires expert interpretation.

## Proposed Solution

OceanTrace automates oil spill detection using a U-Net deep learning segmentation model trained on SAR satellite imagery. The system provides:

- **Automated Oil Spill Segmentation** from SAR scenes
- **Prototype Confidence Scoring** (HIGH / MEDIUM / LOW)
- **Risk Level Classification**
- **Prototype Drift Trajectory Simulation**
- **Interactive Streamlit Dashboard** for rapid decision support

---

## Architecture

```
Satellite SAR Scene
        ↓
Preprocessing (Median Filter + Min-Max Normalization)
        ↓
U-Net Segmentation Model (256×256 patches)
        ↓
Binary Spill Mask + Probability Map
        ↓
Verification Engine (Rule-Based Confidence)
        ↓
Area Estimation + Risk Level
        ↓
Prototype Drift Tracking
        ↓
Streamlit Decision-Support Dashboard
```

---

## Dataset

- **Source**: Oil Spill Segmentation Dataset — Zenodo (https://zenodo.org/records/4672426)
- **Satellite Type**: Sentinel-1 C-Band SAR (VV/VH Polarization)
- **Training Scenes**: 14 large-scale SAR rasters (~2000×5000 px average)
- **Test Scenes**: 7 SAR rasters
- **Pre-extracted Patches**: 21,744 train + 7,249 val (256×256 px, 90-pixel stride)
- **Mask Type**: Binary segmentation (0=Background, 1=Oil Spill)

---

## Preprocessing

- **SAR Speckle Reduction**: Conservative Median Filter (kernel_size=3, configurable)
- **Normalization**: Per-image Min-Max scaling to [0.0, 1.0]
- **Mask Handling**: Nearest-neighbor interpolation only (categorical binary preserved)
- **Augmentation**: HorizontalFlip, VerticalFlip, RandomRotate90 (synchronized image+mask)

---

## AI Model

- **Architecture**: U-Net (Lightweight Hackathon Variant)
- **Input**: 3-channel SAR patch [3, 256, 256]
- **Output**: Binary segmentation logit map [1, 256, 256]
- **Loss**: BCE + Dice Loss (50/50 weighted)
- **Optimizer**: Adam
- **Device**: CUDA if available, else CPU

---

## Training

```bash
python src/train.py --epochs 5 --batch-size 16 --lr 0.001
```

Best model checkpoint automatically saved to: `models/checkpoints/best_unet_model.pth`

---

## Evaluation

```bash
python src/evaluate.py
```

Metrics reported: IoU, Dice Coefficient, Precision, Recall  
Visual comparisons saved to `outputs/visualizations/`

---

## Prediction

```bash
python src/predict.py --image path/to/your/image.tif
```

Results saved to `outputs/predictions/`

---

## Dashboard

```bash
streamlit run app/app.py
```

Open browser at: `http://localhost:8501`

- Upload any SAR satellite image (`.tif`, `.png`, `.jpg`)
- Click **RUN DETECTION**
- View prediction mask, overlay, confidence, risk level, and drift simulation

---

## Limitations

- Prototype/hackathon model — not validated for operational maritime incident response
- Model trained on limited dataset (21 scenes); accuracy improves significantly with more data
- Drift tracking uses prototype wind-based calculation, not real-time ocean current data
- No GPU training in current environment (CPU-only); training speed is limited
- Physical area calculation not available without verified pixel-to-km² metadata

---

## Future Scope

- CUDA GPU training on full dataset for significantly higher accuracy
- Temporal multi-scene analysis for real drift tracking
- Integration with Copernicus/CMEMS real-time ocean current API
- OpenSARShip vessel exclusion layer
- Confidence calibration with Platt scaling

---

## Installation

```bash
# Use Python 3.11
pip install -r requirements.txt

# Train the model first
python src/train.py --epochs 5

# Then launch the dashboard
streamlit run app/app.py
```
