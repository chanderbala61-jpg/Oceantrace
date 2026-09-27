# OceanTrace GitHub Migration & Pre-Deployment Audit

**Repository:** `https://github.com/chanderbala61-jpg/Oceantrace`  
**Problem Statement ID:** 26143 (SIH 2026)  
**Lead Integration Engineer:** Senior Integration Engineer  
**Date:** September 28, 2026  
**Old Commit SHA:** `a36080c1671930abc48f4c2d44a9450222be0523`  
**Backup Tag:** `legacy-pre-deployment`  
**Backup Branch:** `legacy/old-version`  

---

## 1. Executive Summary

A comprehensive architectural and forensic comparison was performed between the legacy version hosted on GitHub (`https://github.com/chanderbala61-jpg/Oceantrace` at commit `a36080c1`) and the current validated local OceanTrace deployment package.

The legacy GitHub repository contains an obsolete prototype based on a toy single-channel UNet model (`best_unet_model_fast.pth`, 10.1 MB) with ad-hoc heuristics, lacking dual-polarization SAR backscatter support, transformer-based segmentation, deterministic RK4 drift hindcasting, multi-factor AIS trajectory correlation, false-positive linkage safeguards, and locked test evaluation metrics.

The current local repository represents the **authoritative source of truth**, featuring the validated SegFormer-B0 architecture (`best.pth`, SHA-256 `cd648dafe044062a69760e71051f06830f62f7857bec42e5ea162400a20648d6`), locked test results (IoU: 0.716094 across 179 scenes), 69/69 passing tests, and the 8-module Streamlit operational dashboard (`app.py`).

---

## 2. Structural Comparison

### 2.1 Old GitHub Repository (`a36080c1`)
```
Oceantrace (Legacy)
├── .gitignore
├── README.md                          # Obsolete prototype description
├── requirements.txt                  # Minimal unpinned prototype dependencies
├── run_pipeline.py                   # Legacy UNet CLI script
├── test_pipeline.py                  # Legacy synthetic test
├── app/
│   └── app.py                        # Legacy Streamlit prototype with uncalibrated mock gauges
├── docs/
│   ├── architecture.md               # Obsolete architecture diagrams
│   └── dataset.md                    # Preliminary dataset notes
├── models/
│   └── checkpoints/
│       ├── .gitkeep
│       ├── best_unet_model_fast.pth  # Obsolete 10.1 MB toy UNet model
│       ├── train_history.json
│       └── train_history_fast.json
└── src/
    ├── dataset.py, evaluate.py, model.py, predict.py
    ├── preprocessing.py, tracking.py, train.py, train_fast.py
    └── utils.py, verification.py
```

### 2.2 Current Validated Local Repository (Source of Truth)
```
OceanTrace (Current Validated Version)
├── app.py                            # Production Streamlit UI (8 Navigation Sections + Folium Maps)
├── config.yaml                       # Machine-independent declarative configuration
├── .env.example                      # Production deployment environment template
├── requirements.txt                  # Fully audited, verified production dependencies
├── Dockerfile                        # Production container definition (Python 3.11-slim)
├── .dockerignore                     # Container build optimization
├── .gitignore                        # Strict credential and large-data barrier
├── README.md                         # Authoritative SIH 2026 documentation & locked metrics
├── src/
│   ├── inference.py                  # Frozen SegFormer loader, radiometric normalization, tiled inference
│   ├── models_exp2.py                # 2-channel SegFormer-B0 builder (mit_b0, 3,712,833 parameters)
│   ├── characterization.py           # Connected components, centroid lat/lon, WGS-84 polygon, area km²
│   ├── look_alike.py                 # Damping contrast, edge sharpness, ambiguity scoring
│   ├── pipeline.py                   # Unified OceanTracePipeline orchestrator & InvestigationRecord
│   ├── dashboard.py                  # High-res diagnostic PNG dashboard & HTML report generator
│   ├── hindcasting/                  # RK4 retro-trajectory integration + Direct baseline fallback
│   └── ais/                          # Multi-factor scoring (0.35/0.30/0.20/0.15) & false-detection link
├── tests/                            # 69 Automated unit & integration tests (100% Passing)
├── outputs/
│   ├── checkpoints/best.pth          # LOCKED SegFormer-B0 checkpoint (SHA-256 cd648daf...)
│   ├── experiment2_test_evaluation/  # LOCKED benchmark evaluation artifacts (179 scenes, IoU 0.716)
│   ├── ais_false_detection_link/     # 58-event linkage report, summaries, and correlation maps
│   ├── investigations/               # Scene 00955 demo data, dashboard PNG, and HTML reports
│   ├── pre_deployment_audit/         # Inventories, risk register, and final readiness audit
│   └── deployment/                   # Deployment records, configuration, and audit reports
└── scripts/
    ├── run_pre_deployment_demo.py    # Reproducible dual-scenario CLI demonstration runner
    └── evaluate_experiment2_test.py  # Locked test evaluation harness
```

---

## 3. Subsystem Forensic Comparison

| Subsystem | Legacy GitHub Repository (`a36080c1`) | Current Validated Local Version | Migration Action |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | Simple 4-stage UNet (`best_unet_model_fast.pth`) | **SegFormer-B0 (`mit_b0`)**, 2-channel SAR input ($3.7\text{M}$ params) | **REPLACE ENTIRELY** |
| **Model Weights** | `models/checkpoints/best_unet_model_fast.pth` (10.1 MB) | `outputs/checkpoints/best.pth` (44.8 MB, SHA-256 verified) | **DEPLOY VALIDATED MODEL** |
| **Test Performance** | Unverified toy accuracy claims | **IoU: 0.716094, Dice: 0.834563** (179 scenes locked) | **ENFORCE LOCKED RESULTS** |
| **SAR Channels** | Single-band grayscale approximation | **Dual-polarization SAR (VH + VV dB)** with radiometric normalization | **REPLACE ENTIRELY** |
| **Look-Alike Filter** | Threshold heuristic | **Damping contrast ($\Delta \sigma^0$) + edge sharpness + wind regime** | **REPLACE ENTIRELY** |
| **Drift Modeling** | Linear velocity extrapolation | **4th-order Runge-Kutta (RK4)** kinematic retro-drift + uncertainty | **REPLACE ENTIRELY** |
| **AIS Correlation** | Simple Euclidean distance | **Segment CPA + Multi-Factor Scoring (0.35/0.30/0.20/0.15)** | **REPLACE ENTIRELY** |
| **Attribution Policy**| Ambiguous vessel blame | **Strict non-accusatory terminology: Candidate Vessel Association** | **REPLACE ENTIRELY** |
| **False Detections** | Unhandled | **Evaluates 56 historical FP scenes; enforces zero temporal AIS overlap**| **REPLACE ENTIRELY** |
| **User Interface** | Single-page script (`app/app.py`) | **8-section interactive Streamlit app (`app.py`) + Folium maps** | **REPLACE ENTIRELY** |
| **Automated Tests** | 1 monolithic script | **69 unit & integration tests in `tests/` (100% passing)** | **REPLACE ENTIRELY** |

---

## 4. Migration Plan: Action Ledger

### 4.1 Files to Preserve via Legacy Tag/Branch:
- The entire existing Git history ending at commit `a36080c1671930abc48f4c2d44a9450222be0523` will be tagged as:
  - **Tag:** `legacy-pre-deployment`
  - **Branch:** `legacy/old-version`

### 4.2 Files to Replace in Working Tree:
- `README.md` -> Replace with production SIH 2026 README.
- `requirements.txt` -> Replace with audited production requirements.
- `.gitignore` -> Replace with comprehensive security and data isolation rules.
- `app/` -> Replace legacy directory with root `app.py`.
- `src/` -> Replace entire directory with validated modular architecture.

### 4.3 Files to Remove from Active Branch:
- `models/checkpoints/best_unet_model_fast.pth` (Obsolete toy model).
- `models/checkpoints/train_history.json` and `train_history_fast.json`.
- `run_pipeline.py` and `test_pipeline.py` (Obsolete scripts).
- `docs/architecture.md` and `docs/dataset.md` (Superseded by updated `README.md` and audit reports).

### 4.4 Files That Must NEVER Be Committed:
- `data/raw/` or full $40+\text{ GB}$ Sentinel-1 GeoTIFF archives.
- Raw training splits or uncompressed images/masks datasets.
- Secrets, credentials, private API tokens, `.env`.
- Windows developer absolute paths (`C:\Users\bala\...`).
- Intermediate training checkpoints (`chunk_001.pth` through `chunk_042.pth`, `latest.pth`).
- OS-specific artifacts (`Thumbs.db`, `.DS_Store`, `desktop.ini`).

---

## 5. Security & Provenance Audit

- **Absolute Path Check:** Passed. Verified zero occurrences of `C:\Users\bala\` in deployed code.
- **Credential Check:** Passed. Verified zero exposed secrets, passwords, or tokens.
- **Model Checksum Check:** Passed. `outputs/checkpoints/best.pth` has SHA-256 `cd648dafe044062a69760e71051f06830f62f7857bec42e5ea162400a20648d6`.
