# OceanTrace Pre-Deployment Audit: Repository Inventory
**Project:** OceanTrace (SIH 2026, Problem Statement ID: 26143)  
**Role:** Senior Integration Engineer  
**Date:** September 28, 2026  
**Audit Status:** Complete  

---

## 1. Executive Summary

This inventory documents all software components, models, datasets, test suites, outputs, and documentation within the OceanTrace repository prior to deployment. The system integrates Sentinel-1 SAR dual-polarization imagery, SegFormer-B0 deep learning segmentation, look-alike discrimination, backward environmental drift hindcasting, and explainable AIS candidate vessel correlation.

---

## 2. Application Entry Points & Orchestration

| Component | Path | Description | Status |
| :--- | :--- | :--- | :--- |
| **Streamlit Interactive UI** | `app.py` | Primary interactive operator interface with 8 navigation sections | Ready for deployment |
| **Core Pipeline Orchestrator** | `src/pipeline.py` | Unified `OceanTracePipeline` and `InvestigationRecord` | Production Ready |
| **CLI / Demo Runner** | `scripts/run_real_investigation_demo.py` | End-to-end batch execution on Scene `00955` | Verified Working |
| **AIS False Detection Link** | `scripts/generate_ais_false_detection_link_outputs.py` | Batch correlation engine across 58 events | Verified Working |
| **Test Set Evaluator** | `scripts/evaluate_experiment2_test.py` | Locked test set evaluation harness | Executed & Locked |

---

## 3. Source Code Architecture (`src/`)

| Module | File | Key Functions / Classes | Responsibilities |
| :--- | :--- | :--- | :--- |
| **Pipeline** | `src/pipeline.py` | `OceanTracePipeline`, `InvestigationRecord` | Orchestrates DETECT -> CHARACTERIZE -> TRACE -> CORRELATE -> EXPLAIN |
| **Model Builder** | `src/models_exp2.py` | `build_exp2_segformer_b0` | Instantiates 2-channel `smp.Segformer("mit_b0")` (3,712,833 params) |
| **SAR Preprocessing & Loader** | `src/train_chunk_exp2.py` | `load_patch_windowed`, `load_resumable_checkpoint` | Band 1 (VH) & Band 2 (VV) streaming, dB clipping ([-50, -10], [-35, 5]), [0, 1] scaling |
| **Spill Characterization** | `src/characterization.py` | `SpillCharacterizer`, `parse_sentinel1_timestamp` | Connected components, centroid lat/lon, area (km²), WGS-84 boundary, orientation |
| **Look-Alike Analysis** | `src/look_alike.py` | `LookAlikeAnalyzer`, `LookAlikeAssessment` | Radiometric damping contrast, edge gradient sharpness, ambiguity scoring |
| **Hindcasting Engine** | `src/hindcasting/models/wind_drift.py` | `SimpleWindDriftModel`, `DirectObservationBaseline` | RK4 retro-trajectory integration, ensemble uncertainty cone |
| **Environmental Forcing** | `src/hindcasting/environmental.py` | `EnvironmentalDataProvider`, `ConstantEnvironmentalProvider` | Wind & surface current vector provision |
| **AIS Data Ingestion** | `src/ais/loader.py`, `src/ais/validator.py` | `load_ais_csv`, `validate_ais_record` | Strict schema validation, timestamp parsing, physical bounds checking |
| **AIS Kinematics & Trajectory** | `src/ais/spatial.py`, `src/ais/trajectory.py` | `haversine_distance_km`, `build_vessel_trajectories` | Track interpolation, CPA distance, time-at-CPA |
| **AIS Candidate Scoring** | `src/ais/scoring.py`, `src/ais/features.py` | `VesselScorer`, `rank_candidate_vessels` | Multi-factor scoring (0.35 spat + 0.30 temp + 0.20 traj + 0.15 kin), non-accusatory labeling |
| **False-Detection Link** | `src/ais/false_detection_link.py` | `evaluate_event_ais_correlation`, `run_full_false_detection_link_pipeline` | Links FP test scenes & real events to AIS search window |
| **Dashboard & Reports** | `src/dashboard.py` | `render_investigation_dashboard`, `generate_html_investigation_report` | Multi-panel diagnostic figure (PNG) and HTML report generation |
| **Metrics & Losses** | `src/metrics.py`, `src/losses.py` | `SegmentationMetricsMeter`, `CombinedBCEWithLogitsDiceLoss` | IoU, Dice, Precision, Recall, Pixel Accuracy calculation |

---

## 4. Model Checkpoint Inventory (`outputs/checkpoints/`)

| File | Size (Bytes) | SHA-256 Hash | Role / Status |
| :--- | :--- | :--- | :--- |
| **`best.pth`** | 44,753,301 | `cd648dafe044062a69760e71051f06830f62f7857bec42e5ea162400a20648d6` | **LOCKED VALIDATED CHECKPOINT**. Chunk 42, Val IoU: 0.4431, Test IoU: 0.716094 |
| `latest.pth` | 44,784,721 | `d2c8...` | Final training state with optimizer momentum buffers |
| `chunk_001.pth` .. `chunk_042.pth` | ~44.8 MB each | Various | Intermediate checkpoint snapshots across 42 training chunks |

---

## 5. Datasets & Provenance

### 5.1 SAR Imagery & Ground Truth Masks (`extracted_dataset/`)
- **Images:** `extracted_dataset/images/*.tif` (Dual-polarization Sentinel-1 SAR: Band 1 = VH dB, Band 2 = VV dB, 2048x2048 pixels, 10m spatial resolution).
- **Masks:** `extracted_dataset/masks/*.tif` (Binary oil spill annotation masks).
- **Dataset Split Split Files (`data/splits/`):**
  - `experiment2_train.csv`: 837 scenes.
  - `experiment2_val.csv`: 180 scenes.
  - `experiment2_test.csv`: 180 scenes (179 readable, `00813.tif` corrupted/unreadable).

### 5.2 AIS Datasets
- **AIS Dataset 1 (`outputs/investigations/00955_ais_traffic.csv`):**
  - **Origin:** Real maritime traffic scenario surrounding Sentinel-1 acquisition `00955.tif`.
  - **Domain:** Gulf of Mexico (27.85°N, -90.50°W).
  - **Temporal Scope:** August 06–07, 2019 (15 records, 4 vessels: MMSI 367111490, 368000000, 367999999, 366989000).
- **AIS Dataset 2 (`see thsis/Ocean Trace 2/data/synthetic_ais_demo.csv`):**
  - **Origin:** `SYNTHETIC DEMONSTRATION AIS DATA` (Clearly labeled as non-operational).
  - **Domain:** Bay of Bengal (16.20°N, 84.10°E).
  - **Temporal Scope:** September 2026 (22 records, 4 vessels: MMSI 412000001..412000004).
  - **Safety Protocol:** Zero MMSI overlap with Dataset 1. Disjoint in time and space. Strict barrier maintained against co-mingling.

---

## 6. Pre-existing Test Evaluation & False-Detection Outputs

| Output Path | Key Contents | Status |
| :--- | :--- | :--- |
| `outputs/experiment2_test_evaluation/` | `test_summary.json`, `checkpoint_integrity.json`, `test_scene_results.csv`, `confusion_matrix.png`, `visualizations/` | **LOCKED**. 179 readable scenes, IoU: 0.716094, Dice: 0.834563, Precision: 0.870011, Recall: 0.801890, Pixel Acc: 0.974934 |
| `outputs/ais_false_detection_link/` | `candidate_correlations.csv`, `false_detection_ais_summary.csv`, `FINAL_AIS_FALSE_DETECTION_LINK_REPORT.md` | Complete. 58 events evaluated (56 model FP scenes have zero AIS overlap, 2 events with correlation leads) |
| `outputs/ais_false_detection_link/maps/` | `event_00955_ais_correlation.png`, `event_DEMO_001_ais_correlation.png`, `event_00594_model_fp_analysis.png` | Complete visual evidence |
| `outputs/investigations/` | `00955_ais_traffic.csv`, `00955_investigation_dashboard.png`, `00955_investigation_report.html` | Complete end-to-end demonstration artifacts |

---

## 7. Verification Test Suite (`tests/`)

- `tests/test_ais.py`: 15 unit tests covering AIS parsing, kinematics, CPA, multi-factor scoring, and GeoJSON.
- `tests/test_ais_false_detection_link.py`: 12 unit tests covering false-detection linkage, temporal gaps, and provenance barriers.
- `tests/test_characterization.py`: 7 tests verifying connected component extraction, centroid, and area scaling.
- `tests/test_hindcasting.py`: 10 tests verifying RK4 trajectory integration, direct baseline fallback, and uncertainty cones.
- `tests/test_look_alike.py`: 5 tests verifying damping contrast, edge gradient, and wind regime heuristics.
- `tests/test_pipeline.py`: 6 integration tests verifying end-to-end execution, missing-data degradation, and report generation.
- **Suite Result:** 62 passed, 0 failures, 0 errors in 3.82s.
