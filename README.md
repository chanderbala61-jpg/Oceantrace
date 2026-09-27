# OceanTrace: Maritime Oil Spill Intelligence & AIS Vessel Correlation

**Smart India Hackathon (SIH) 2026**  
**Problem Statement ID:** 26143  
*“Leveraging satellite imagery to determine Oil spills at sea along with AIS data correlations to identify vessel responsible for the spill.”*  

---

## 1. Executive Overview

**OceanTrace** is an end-to-end maritime environmental intelligence platform that unifies Sentinel-1 Synthetic Aperture Radar (SAR) satellite imagery with Automated Identification System (AIS) vessel transponder telemetry. The system automates:

$$\text{Sentinel-1 SAR} \rightarrow \text{SegFormer-B0 Segmentation} \rightarrow \text{Look-Alike Filtering} \rightarrow \text{Morphometric Characterization} \rightarrow \text{RK4 Drift Hindcasting} \rightarrow \text{AIS Spatiotemporal Correlation} \rightarrow \text{Explainable Investigation Lead}$$

```
                Sentinel-1 Dual-Polarization SAR (VH + VV dB)
                                    │
                                    ▼
                     SAR Radiometric Preprocessing
                     (dB Clipping & Min-Max Scaling)
                                    │
                                    ▼
                SegFormer-B0 Deep Learning Segmentation
                       (Frozen Checkpoint: best.pth)
                                    │
                                    ▼
                      Binary Oil-Spill Slick Mask
                                    │
                                    ▼
                    Radiometric Look-Alike Analysis
                    (Damping Contrast & Edge Gradient)
                                    │
                                    ▼
                 Spill Characterization & Morphometry
                 (Surface Area, Centroid Lat/Lon, BBox)
                                    │
                                    ▼
               Backward Environmental Drift Hindcasting
                (4th-Order Runge-Kutta + Uncertainty)
                                    │
                                    ▼
               AIS Spatiotemporal Trajectory Correlation
               (Haversine Distance, Time Window, CPA)
                                    │
                                    ▼
                   Multi-Factor Candidate Vessel Scoring
                  (0.35 Spatial + 0.30 Temp + 0.20 Traj + 0.15 Kin)
                                    │
                                    ▼
             Formal Investigation Dossier & Interactive UI
                   (Streamlit Dashboard, HTML & PNG Exports)
```

---

## 2. System Architecture & Modules

The repository is structured into modular, production-ready subsystems under `src/`:

```
exp 1/
├── app.py                            # Streamlit Interactive Demonstration Dashboard (8 Modules)
├── config.yaml                       # Declarative, machine-independent configuration
├── requirements.txt                  # Clean, verified production dependencies
├── src/
│   ├── inference.py                  # Frozen SegFormer-B0 loader, radiometric scaling, tiled inference
│   ├── models_exp2.py                # 2-channel SegFormer-B0 architecture (mit_b0, 3.7M parameters)
│   ├── characterization.py           # Connected component extraction, centroid, area km², polygon
│   ├── look_alike.py                 # Damping contrast, edge sharpness, ambiguity scoring
│   ├── pipeline.py                   # Unified OceanTracePipeline orchestrator & InvestigationRecord
│   ├── dashboard.py                  # Matplotlib diagnostics rendering & HTML report generation
│   ├── hindcasting/                  # Environmental drift modeling engine
│   │   ├── models/wind_drift.py      # 4th-order Runge-Kutta (RK4) retro-trajectory integrator
│   │   ├── models/direct_baseline.py # Level 0 Direct Observation Baseline fallback
│   │   └── environmental.py          # Wind & ocean current vector providers
│   └── ais/                          # AIS spatial-temporal correlation engine
│       ├── loader.py & validator.py  # Schema ingestion & physical boundary validation
│       ├── spatial.py & temporal.py  # Haversine distance, segment CPA, time window filtering
│       ├── trajectory.py             # Vessel track reconstruction & interpolation
│       ├── scoring.py & features.py  # 4-factor scoring & non-accusatory classification
│       ├── false_detection_link.py   # Multi-event evaluation & temporal barrier enforcement
│       └── visualization.py          # Folium & GeoJSON visual export utilities
├── tests/                            # 69 Automated unit & integration tests (100% Passing)
├── outputs/
│   ├── checkpoints/best.pth          # LOCKED & FROZEN validated SegFormer-B0 checkpoint
│   ├── experiment2_test_evaluation/  # LOCKED official benchmark evaluation results
│   ├── ais_false_detection_link/     # 58-event linkage report, summaries, and correlation maps
│   ├── investigations/               # Scene 00955 demo data, dashboard PNG, and HTML reports
│   └── pre_deployment_audit/         # Pre-deployment inventories, risk register, and readiness report
└── scripts/
    ├── run_pre_deployment_demo.py    # Reproducible dual-scenario CLI validation runner
    └── evaluate_experiment2_test.py  # Locked test evaluation harness
```

---

## 3. Currently Implemented & Validated Features

### 3.1 Deep Learning SAR Segmentation (SegFormer-B0)
- **Model:** SegFormer-B0 with `mit_b0` hierarchical transformer encoder and lightweight all-MLP decoder.
- **Input Channels:** 2 genuine dual-polarization channels (Band 1 = $\sigma^0_{\text{VH}}\text{ dB}$, Band 2 = $\sigma^0_{\text{VV}}\text{ dB}$).
- **Parameters:** 3,712,833 total parameters (**0 trainable parameters in deployment; model is frozen**).
- **Inference Mode:** Memory-efficient patch-windowed inference with fixed locked decision threshold $p \ge 0.50$.
- **Checkpoint Location:** `outputs/checkpoints/best.pth` (SHA-256: `cd648dafe044062a69760e71051f06830f62f7857bec42e5ea162400a20648d6`).

### 3.2 Locked Test Evaluation Benchmark
Evaluated strictly on `data/splits/experiment2_test.csv` (180 scenes total; `00813.tif` explicitly quarantined as unreadable):

| Metric | Locked Benchmark Value | Verification Status |
| :--- | :--- | :--- |
| **Readable Test Scenes** | **179 scenes** | Evaluated without post-hoc tuning |
| **Unreadable Scene** | **00813.tif** | Quarantined (`status: unreadable`) |
| **Intersection over Union (IoU)** | **0.716094** | **LOCKED & VERIFIED** |
| **Dice Coefficient (F1-Score)** | **0.834563** | **LOCKED & VERIFIED** |
| **Precision** | **0.870011** | **LOCKED & VERIFIED** |
| **Recall** | **0.801890** | **LOCKED & VERIFIED** |
| **Pixel Accuracy** | **0.974934** | **LOCKED & VERIFIED** |
| **True Positive Pixels (TP)** | 1,483,371 | Immutable |
| **True Negative Pixels (TN)** | 21,390,413 | Immutable |
| **False Positive Pixels (FP)** | 221,631 | Immutable |
| **False Negative Pixels (FN)** | 366,473 | Immutable |

### 3.3 Look-Alike & Ambiguity Filtering
- Radiometric damping contrast ($\Delta \sigma^0 = \text{Mean}_{\text{sea}} - \text{Mean}_{\text{slick}}$).
- Edge sharpness gradient ($\text{dB/pixel}$) via Sobel/Scharr filtering.
- Wind regime heuristics (low-wind calm water $<3\text{ m/s}$ flagged for natural biogenic look-alikes).

### 3.4 Backward Environmental Drift Hindcasting
- **Solver:** Deterministic 4th-order Runge-Kutta (RK4) kinematic integration.
- **Forcing:** Sea-surface wind vectors ($u, v\text{ m/s}$) with standard 3% empirical windage factor.
- **Uncertainty Region:** Backward-expanding ensemble uncertainty circle ($\pm \text{km}$) accounting for turbulent diffusion and wind variance.
- **Graceful Fallback:** Automatic degradation to Level 0 Direct Observation Baseline if environmental data is unavailable.
- **Naming Standard:** Always labeled as *"Probable Origin Estimate"*, never *"Confirmed Release Location"*.

### 3.5 AIS Correlation & Multi-Factor Scoring
- Filters vessel transponder broadcasts within spatiotemporal search corridor (default $50\text{ km}$, $-24\text{h}$ to $+6\text{h}$).
- Interpolates tracks to compute Closest Point of Approach (CPA) distance and temporal offset.
- **Scoring Formula:**
  $$\text{Score} = 0.35 \times S_{\text{spatial}} + 0.30 \times S_{\text{temporal}} + 0.20 \times S_{\text{trajectory}} + 0.15 \times S_{\text{kinematics}}$$
- **Non-Accusatory Classification:** Candidates are designated as *"Potential Spatiotemporal Association"*, *"Weak Association"*, or *"No Sufficient AIS Evidence"*. Never *"guilty"* or *"confirmed polluter"*.

### 3.6 False-Detection Linkage & Provenance Safeguards
- Evaluated across 58 events (2 operational/demo events + 56 model test false-positive scenes).
- The 56 test scenes (acquisitions from 2015–2018) have zero temporal overlap with available AIS data (2019/2026).
- The system enforces a strict temporal barrier: zero vessel blame is assigned to historical FP scenes, outputting: *"No sufficient AIS evidence was available for this event within the configured search window."*
- Complete isolation maintained between AIS Dataset 1 (Gulf of Mexico, Aug 2019) and AIS Dataset 2 (`SYNTHETIC DEMONSTRATION AIS DATA`, Bay of Bengal, Sep 2026).

---

## 4. Installation & Local Execution

### 4.1 Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit)
- Modern multi-core CPU (GPU optional; CUDA supported automatically if available)
- 4 GB RAM minimum (8 GB recommended)

### 4.2 Setup
```bash
# Clone the repository
git clone https://github.com/your-org/oceantrace.git
cd oceantrace

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# Install validated dependencies
pip install -r requirements.txt
```

### 4.3 Running the Streamlit Application
```bash
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

### 4.4 Running Headless in Cloud / Container
```bash
python -m streamlit run app.py --server.port 8501 --server.headless true --server.enableCORS false
```

---

## 5. End-to-End Demonstration Scenarios

OceanTrace provides a reproducible CLI demonstration runner executing the two primary validation scenarios:

```bash
python scripts/run_pre_deployment_demo.py
```

### Demo Scenario 1: Scene 00955 (Real S-1 SAR Slick & AIS Dataset 1)
- **Domain:** Gulf of Mexico ($27.85^\circ\text{N}, -90.50^\circ\text{W}$).
- **Acquisition Time:** 2019-08-07 00:25:51 UTC.
- **Workflow:** S-1 Preprocessing $\rightarrow$ SegFormer-B0 Inference (3.29s) $\rightarrow$ True Oil Spill confirmed ($4.53\text{ dB}$ damping contrast) $\rightarrow$ Area $2.273\text{ km}^2$ $\rightarrow$ RK4 Hindcast Origin ($27.828^\circ\text{N}, -90.545^\circ\text{W} \pm 8.0\text{ km}$) $\rightarrow$ AIS Correlation finds 2 ranked candidate leads:
  - Rank #1: `TEST_TANKER_ALPHA` (Score: 0.97, CPA: 1.7 km, dt: +0.0h, Highly Compatible Candidate).
  - Rank #2: `TEST_TUG_DELTA` (Score: 0.57, CPA: 22.3 km, dt: +0.0h, Moderately Compatible Candidate).
- **Exports:** High-resolution multi-panel dashboard PNG and formal HTML investigation report.

### Demo Scenario 2: Scene 00594 (Model FP Limitation Case - Scientific Integrity)
- **Acquisition Time:** 2017-06-25 14:30:00 UTC.
- **Workflow:** Model generates FP pixels with weak contrast ($2.70\text{ dB} < 3.5\text{ dB}$).
- **Integrity Safeguard:** System detects that event is from 2017 and local AIS coverage is unavailable for that period.
- **Result:** Reports *"No sufficient AIS evidence was available"* and attributes **zero** vessels. Zero fabricated matches.

---

## 6. Verification Test Suite

Run the full automated test suite (69 tests):

```bash
python -m unittest discover tests
```

Expected result:
```
Ran 69 tests in ~11.8s
OK
```

---

## 7. Operational & Scientific Limitations

1. **Deterministic Kinematic Drift Solver:** OceanTrace uses a 2D RK4 kinematic wind-drift model with an empirical 3% windage factor. Complex 3D oil weathering (emulsification, evaporation, vertical droplet dispersion) via OpenDrift/OpenOil is not currently implemented.
2. **Temporal Coverage Constraint:** AIS correlation requires historical transponder broadcasts covering the exact acquisition timeframe and geographic bounding box. When coverage is absent, no attribution can be performed.
3. **Investigative Nature:** AIS candidates represent spatiotemporal corridor leads. Presence within the corridor does not constitute proof of deliberate discharge. Legal determination requires physical coast guard inspection or onboard sampling.
4. **Diagnostic Metrics:** Any scene-level IoU/Dice displayed during demo execution is purely for diagnostic comparison against companion masks and does not alter the locked test benchmark.

---

## 8. Future Extensions (Roadmap)

The following capabilities represent planned post-deployment enhancements:
- **Live Reanalysis API Integration:** Direct streaming from Copernicus Marine Service (CMEMS) and NOAA GFS for automated real-time wind and ocean current ingestion.
- **Operational Satellite AIS Feeds:** Integration with live global terrestrial/satellite AIS aggregators (Spire, AISHub, MarineTraffic).
- **3D Hydrodynamic Weathering:** Integration of OpenDrift/OpenOil transport modeling with turbulent dissolution physics.
- **Counterfactual Attribution Testing:** Forward trajectory simulation from all candidate vessels to verify which vessel track geometrically reproduces the observed slick boundary.
- **Multi-Mission SAR Support:** Ingestion adapters for NISAR, RADARSAT Constellation Mission (RCM), and TerraSAR-X.

---

## 9. Legal & Ethical Disclaimer

> **DISCLAIMER:** OceanTrace is an environmental decision-support tool designed to provide objective, explainable investigative leads to maritime authorities and environmental protection agencies. Spatiotemporal proximity derived from AIS broadcasts does not prove vessel culpability or intentional discharge. OceanTrace does not establish legal liability.

---

## 10. Pre-Deployment Acceptance Sign-Off

- **Lead Integration Engineer:** Senior Integration Engineer
- **Test Suite Status:** 69 / 69 Tests Passed (0 Failures, 0 Errors)
- **Model Checkpoint Status:** Frozen & Verified (`cd648dafe044062a69760e71051f06830f62f7857bec42e5ea162400a20648d6`)
- **Locked Benchmark IoU:** 0.716094 (179 scenes)
- **Pre-Deployment Readiness Determination:** **PRE-DEPLOYMENT READY**
