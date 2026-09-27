# OceanTrace Pre-Deployment Audit: Risk Register
**Project:** OceanTrace (SIH 2026, Problem Statement ID: 26143)  
**Role:** Senior Integration Engineer  
**Date:** September 28, 2026  
**Status:** All Identified Risks Mitigated with Safeguards  

---

## 1. Risk Matrix Overview

| Risk ID | Category | Risk Description | Severity | Likelihood | Residual Risk | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RSK-01** | Model Integrity | Accidental modification or weight overwriting of frozen `best.pth` | Critical | Low | Negligible | **MITIGATED** |
| **RSK-02** | Evaluation Bias | Data leakage or threshold tuning on test dataset | Critical | Low | Zero | **MITIGATED** |
| **RSK-03** | Legal / Ethics | Misidentifying candidate vessel as "guilty polluter" or legally liable | High | Medium | Very Low | **MITIGATED** |
| **RSK-04** | Data Provenance | Inadvertent co-mingling of Gulf of Mexico AIS (2019) and Bay of Bengal Synthetic AIS (2026) | High | Medium | Zero | **MITIGATED** |
| **RSK-05** | False Attribution | Attempting to match 2015–2018 model FP scenes to 2019/2026 AIS data | High | High | Zero | **MITIGATED** |
| **RSK-06** | Hardware / OOM | System memory exhaustion when loading large 2048x2048 SAR GeoTIFFs | Medium | Medium | Low | **MITIGATED** |
| **RSK-07** | Environmental Data | Unavailability of real-time wind/current forcing causing pipeline failure | Medium | High | Low | **MITIGATED** |
| **RSK-08** | Input File Robustness| Corrupted, missing, or malformed input TIFF files (e.g., `00813.tif`) | Medium | Medium | Very Low | **MITIGATED** |
| **RSK-09** | Path Portability | Hardcoded Windows developer paths (`C:\Users\bala\...`) breaking remote deployment | High | High | Zero | **MITIGATED** |

---

## 2. In-Depth Risk Analysis & Enforced Mitigations

### RSK-01: Model Checkpoint Modification
- **Hazard:** Re-training, weight alteration, or running optimizer steps during inference or evaluation corrupts the validated SegFormer-B0 baseline.
- **Enforced Safeguards:**
  - `best.pth` has SHA-256 fingerprint: `cd648dafe044062a69760e71051f06830f62f7857bec42e5ea162400a20648d6`.
  - All inference scripts load the checkpoint in `torch.no_grad()` mode with model set to `.eval()`.
  - Pre- and post-execution checksum validations verify zero byte changes.

### RSK-02: Test Dataset Data Leakage & Threshold Tuning
- **Hazard:** Tuning decision thresholds on test data artificially inflates IoU/Dice metrics, violating scientific benchmark rules.
- **Enforced Safeguards:**
  - Locked test evaluation is strictly preserved (179 readable scenes, IoU: 0.716094, Dice: 0.834563, Precision: 0.870011, Recall: 0.801890, Pixel Accuracy: 0.974934).
  - Decision threshold is fixed at $0.5$ without post-hoc optimization.

### RSK-03: Defamation / False Legal Accusations of Maritime Vessels
- **Hazard:** Users or investigators misinterpreting high spatiotemporal proximity as proof of deliberate oil discharge.
- **Enforced Safeguards:**
  - Strict non-accusatory terminology enforced across code, UI, and exported reports: "Candidate Vessel", "Potential Spatiotemporal Association", "Investigation Lead".
  - Mandatory legal disclaimer prominently displayed:
    > *"OceanTrace provides explainable investigative leads based on spatiotemporal and hydrodynamic correlations. It does not establish legal responsibility or prove intentional discharge."*

### RSK-04: AIS Dataset Co-Mingling
- **Hazard:** Combining AIS Dataset 1 (Gulf of Mexico, Aug 2019) with AIS Dataset 2 (Bay of Bengal, Sep 2026) produces phantom trajectories and nonsensical search radii.
- **Enforced Safeguards:**
  - Complete structural and runtime isolation between datasets.
  - AIS Dataset 2 is explicitly labeled in UI and metadata as `SYNTHETIC DEMONSTRATION AIS DATA`.
  - Geographic and temporal bounds checking prevents cross-domain querying.

### RSK-05: False Attribution on Model False-Positive Scenes
- **Hazard:** The 56 test scenes where SegFormer-B0 produced false-positive pixels date from 2015 to 2018. If queried against 2019/2026 AIS data, arbitrary vessels could be falsely linked.
- **Enforced Safeguards:**
  - `src/ais/false_detection_link.py` rigorously validates temporal bounds.
  - When temporal overlap is zero, the system outputs:
    > *"No sufficient AIS evidence was available for this event within the configured search window."*
  - Zero vessel candidates are returned for these 56 scenes.

### RSK-06: Resource Exhaustion & Memory Overflows (OOM)
- **Hazard:** Loading multiple 2048x2048 32-bit floating-point SAR GeoTIFFs (16 MB uncompressed each) repeatedly into memory can cause out-of-memory crashes on systems without dedicated GPUs.
- **Enforced Safeguards:**
  - Streamlit model caching (`@st.cache_resource`) ensures the SegFormer-B0 model is loaded once into memory per server session.
  - Windowed loading (`load_patch_windowed`) and spatial downsampling for visual overlays avoid large intermediate buffers.
  - CPU fallback supported natively when CUDA is unavailable.

### RSK-07: Missing Environmental Forcing Data
- **Hazard:** Live oceanographic reanalysis (ERA5/HYCOM/GFS) APIs may be unreachable, offline, or restricted.
- **Enforced Safeguards:**
  - Multi-tier fallback architecture: If wind/current data is absent, the system degrades gracefully to the Level 0 Direct Observation Baseline, using the observed slick centroid and alerting the operator of degraded origin precision.

### RSK-08: Corrupted or Malformed TIFF Files
- **Hazard:** Unreadable or truncated GeoTIFFs (such as test scene `00813.tif`) causing unhandled exceptions.
- **Enforced Safeguards:**
  - Explicit try-catch wrapping in file ingestion with diagnostic warning banners.
  - Corrupted files are flagged as `status: unreadable` without aborting batch or UI flows.

### RSK-09: Hardcoded Machine-Specific Paths
- **Hazard:** Hardcoded paths like `C:\Users\bala\...` break deployment when run in Linux containers, Docker, cloud VMs, or other developer machines.
- **Enforced Safeguards:**
  - All filepaths refactored to relative paths or resolved relative to `REPO_ROOT` / `config.yaml`.
  - Machine-specific path audit conducted across all active source files.
