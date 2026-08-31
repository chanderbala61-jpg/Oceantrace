# OceanTrace Architecture Overview

OceanTrace is an AI-powered system designed for end-to-end satellite oil spill detection, verification, and drift trajectory tracking.

## System Architecture Pipeline

1. **Data Ingestion & Preprocessing (`src/dataset.py`, `src/preprocessing.py`)**
   - Ingest satellite SAR imagery (e.g., Sentinel-1).
   - Apply speckle noise reduction, normalization, and patch extraction.

2. **AI Segmentation Engine (`src/model.py`, `src/predict.py`)**
   - Deep learning segmentation (U-Net / DeepLabV3+) to detect dark spots and potential oil slicks.

3. **Verification Engine (`src/verification.py`)**
   - Rule-based and contextual filtering to distinguish true oil spills from look-alikes (low wind zones, biogenic slicks).

4. **Drift & Trajectory Tracking (`src/tracking.py`)**
   - Simulate slick displacement based on ocean current and wind data over time.

5. **Dashboard & Alerting UI (`app/app.py`)**
   - Interactive user dashboard for monitoring detected spills, viewing confidence scores, and inspecting tracking vectors.
