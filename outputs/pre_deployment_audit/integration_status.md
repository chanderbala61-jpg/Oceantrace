# OceanTrace Pre-Deployment Audit: Subsystem Integration Status
**Project:** OceanTrace (SIH 2026, Problem Statement ID: 26143)  
**Role:** Senior Integration Engineer  
**Date:** September 28, 2026  
**Status:** All Core Modules Integrated & Verified  

---

## 1. End-to-End Pipeline Integration Map

```
Sentinel-1 SAR (Dual-Pol VH + VV)
        ↓
[1] SAR Preprocessing (dB clipping, min-max scaling, patch-windowing)
        ↓
[2] SegFormer-B0 Inference (mit_b0 2-channel, frozen best.pth)
        ↓
[3] Oil-Spill Segmentation Mask (threshold 0.5, connected components)
        ↓
[4] False-Positive / Look-Alike Analysis (damping contrast, edge sharpness)
        ↓
[5] Spill Characterization (area km², centroid lat/lon, WGS-84 polygon)
        ↓
[6] Hindcasting / Probable-Origin (RK4 drift integration, ensemble cone)
        ↓
[7] AIS Spatial-Temporal Correlation (temporal filter, CPA, segment distance)
        ↓
[8] Candidate Vessel Associations (multi-factor score, non-accusatory)
        ↓
[9] Evidence + Uncertainty Compilation (data provenance, audit trail)
        ↓
[10] Investigation Report & Dashboard (PNG dashboard, HTML report, Streamlit UI)
```

---

## 2. Detailed Stage-by-Stage Integration Status

### Stage 1: SAR Input & Preprocessing
- **Status:** `FULLY IMPLEMENTED`
- **Module:** `src/train_chunk_exp2.py` (`load_patch_windowed`), `scripts/run_real_investigation_demo.py`
- **Implementation:** Extracts dual-polarization SAR channels (Band 1 = VH dB, Band 2 = VV dB) using `rasterio`. Clips extreme values (VH: [-50, -10] dB, VV: [-35, 5] dB) and scales to $[0, 1]$.
- **Hardware Optimization:** Windowed patch loading prevents memory exhaustion on large 2048x2048 scenes.
- **Verification:** Verified across 180 test scenes and live demo on Scene `00955`.

### Stage 2: SegFormer-B0 Model Inference
- **Status:** `FULLY IMPLEMENTED & FROZEN`
- **Module:** `src/models_exp2.py` (`build_exp2_segformer_b0`), `outputs/checkpoints/best.pth`
- **Implementation:** 2-channel `smp.Segformer("mit_b0")` with 3,712,833 trainable parameters. Sigmoid activation with fixed locked threshold 0.5.
- **Integrity Status:** Checkpoint SHA-256 (`cd648dafe044062a69760e71051f06830f62f7857bec42e5ea162400a20648d6`) verified before and after all evaluation passes. Zero retraining or threshold tuning.
- **Locked Test Benchmark:** 179 readable scenes: IoU: 0.716094, Dice: 0.834563, Precision: 0.870011, Recall: 0.801890, Pixel Accuracy: 0.974934.

### Stage 3: Segmentation Mask Extraction
- **Status:** `FULLY IMPLEMENTED`
- **Module:** `src/pipeline.py`, `src/metrics.py`
- **Implementation:** Binary mask generation ($p \ge 0.5$), connected component filtering. Clearly separates model output from ground truth companion masks.

### Stage 4: False-Positive & Look-Alike Analysis
- **Status:** `FULLY IMPLEMENTED`
- **Module:** `src/look_alike.py` (`LookAlikeAnalyzer`), `src/ais/false_detection_link.py`
- **Implementation:** Calculates damping contrast ($\Delta \sigma^0 \text{ dB}$ between background ocean and slick interior), edge sharpness, aspect ratio, and wind regime flags.
- **False-Detection Linkage:** Assesses 56 test scenes with model FP pixels. Verifies that these scenes (2015–2018 acquisitions) have zero AIS temporal overlap with available 2019/2026 data. Correctly reports: *"No sufficient AIS evidence was available for this event within the configured search window."*

### Stage 5: Spill Characterization
- **Status:** `FULLY IMPLEMENTED`
- **Module:** `src/characterization.py` (`SpillCharacterizer`)
- **Implementation:** Extracts spatial centroid (lat/lon), surface area ($\text{km}^2$), perimeter ($\text{km}$), bounding box, polygon boundary (WGS-84 coordinates), and major axis orientation angle.
- **Verification:** Verified via `tests/test_characterization.py` (7/7 passed).

### Stage 6: Environmental Drift Hindcasting
- **Status:** `FULLY IMPLEMENTED (Kinematic Drift + RK4)`
- **Module:** `src/hindcasting/models/wind_drift.py` (`SimpleWindDriftModel`), `src/hindcasting/environmental.py`
- **Implementation:** 4th-order Runge-Kutta (RK4) backward numerical integration. Uses windage factor (default 3%) and surface current vectors. Computes ensemble uncertainty cone ($\pm \text{km}$) expanding backward in time.
- **Clarification:** OpenDrift/OpenOil is *not* used. System uses custom lightweight, highly deterministic RK4 solver.
- **Labeling Standard:** Clearly presented as "Probable Origin Estimate", never "Confirmed Release Location".

### Stage 7: AIS Spatial-Temporal Correlation
- **Status:** `FULLY IMPLEMENTED`
- **Module:** `src/ais/` (`loader.py`, `validator.py`, `spatial.py`, `temporal.py`, `trajectory.py`)
- **Implementation:** Filters vessel broadcasts within temporal window ($t_{\text{spill}} - \Delta t_{\text{back}}$ to $t_{\text{spill}} + \Delta t_{\text{forward}}$). Reconstructs trajectories, calculates Haversine distance, and computes Closest Point of Approach (CPA) to estimated origin and observed slick.
- **Verification:** Verified via `tests/test_ais.py` (15/15 passed).

### Stage 8: Candidate Vessel Associations
- **Status:** `FULLY IMPLEMENTED`
- **Module:** `src/ais/scoring.py` (`VesselScorer`, `rank_candidate_vessels`)
- **Scoring Formula:**
  $$\text{Score} = 0.35 \times S_{\text{spatial}} + 0.30 \times S_{\text{temporal}} + 0.20 \times S_{\text{trajectory}} + 0.15 \times S_{\text{kinematics}}$$
- **Ethical & Legal Safeguards:** Candidates are classified as "Potential Spatiotemporal Association", "Weak Association", or "No Sufficient AIS Evidence". Strictly non-accusatory terminology.

### Stage 9: Evidence & Uncertainty Compilation
- **Status:** `FULLY IMPLEMENTED`
- **Module:** `src/pipeline.py` (`InvestigationRecord`), `src/ais/false_detection_link.py`
- **Implementation:** Documents sensor metadata, acquisition time, assumptions, missing data flags, and CPA telemetry. Distinguishes AIS Dataset 1 (Gulf of Mexico, Aug 2019) from AIS Dataset 2 (Synthetic Demo, Bay of Bengal, Sep 2026).

### Stage 10: Investigation Reporting & Maps
- **Status:** `FULLY IMPLEMENTED`
- **Module:** `src/dashboard.py`, `outputs/ais_false_detection_link/maps/`
- **Implementation:** Produces 300-DPI multi-panel diagnostic PNG dashboards and standalone self-contained HTML reports with embedded CSS styling.

### Stage 11: Streamlit Demonstration Application
- **Status:** `READY FOR DEPLOYMENT`
- **Module:** `app.py`
- **Implementation:** Multi-page sidebar navigation (Dashboard, Spill Detection, Characterization, Hindcast, AIS Investigation, Evidence Summary, Report, System Info) with real-time inference and Folium mapping.
