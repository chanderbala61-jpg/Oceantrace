# OceanTrace Final Pre-Deployment Master Report

**Project:** OceanTrace  
**Competition:** Smart India Hackathon (SIH) 2026  
**Problem Statement ID:** 26143  
*“Leveraging satellite imagery to determine Oil spills at sea along with AIS data correlations to identify vessel responsible for the spill.”*  
**Auditor / Role:** Senior Integration Engineer  
**Date:** September 28, 2026  
**Final Status:** **PRE-DEPLOYMENT READY / GITHUB UPDATED**  

---

## 1. Executive Summary

This report documents the replacement and deployment of the OceanTrace system on the official GitHub repository:
`https://github.com/chanderbala61-jpg/Oceantrace`

The legacy GitHub repository contained an outdated prototype using a 10.1 MB toy UNet model with uncalibrated heuristics. It has been backed up, tagged, and replaced with the current validated, peer-reviewed OceanTrace architecture featuring:
- SegFormer-B0 deep transformer segmentation (frozen `best.pth`, SHA-256 `cd648dafe044062a69760e71051f06830f62f7857bec42e5ea162400a20648d6`).
- Locked test evaluation results: 179 readable scenes, IoU: 0.716094, Dice: 0.834563, Precision: 0.870011, Recall: 0.801890, Pixel Accuracy: 0.974934.
- 4th-order Runge-Kutta (RK4) kinematic retro-drift hindcasting with ensemble uncertainty cones.
- Multi-factor AIS candidate vessel correlation ($0.35\text{ spat} + 0.30\text{ temp} + 0.20\text{ traj} + 0.15\text{ kin}$).
- Strict non-accusatory legal safeguards (*Candidate Vessel Association*, not *Guilty Polluter*).
- 58-event false-detection linkage engine ensuring zero vessel blame on historical false-positive scenes.
- 69/69 automated tests passing with 0 failures and 0 errors.
- 8-module Streamlit operational dashboard (`app.py`) with interactive Folium Leaflet maps.

---

## 2. GitHub Version & Tag Ledger

| State | Reference | Commit SHA | Description |
| :--- | :--- | :--- | :--- |
| **Old GitHub Version** | `refs/heads/main` | `a36080c1671930abc48f4c2d44a9450222be0523` | Legacy toy UNet prototype |
| **Legacy Backup Tag** | `refs/tags/legacy-pre-deployment` | `a36080c1671930abc48f4c2d44a9450222be0523` | **PERMANENT BACKUP TAG ON GITHUB** |
| **Legacy Backup Branch**| `refs/heads/legacy/old-version` | `a36080c1671930abc48f4c2d44a9450222be0523` | **PERMANENT RECOVERABLE BRANCH ON GITHUB** |
| **New Deployment Version**| `refs/heads/main` | `5bfc6f8cfc2a22e77ca114df8106d175a334d903` | Validated production deployment version |

---

## 3. Deployment Artifacts & Integrity Verification

- **Production Dockerfile:** Implemented using `python:3.11-slim`, exposes port 8501, includes minimal C-extension libraries (`libgomp1`, `build-essential`, `curl`), and embeds healthcheck.
- **Docker Ignore (.dockerignore):** Strictly excludes raw $40+\text{ GB}$ archives, intermediate chunk checkpoints (`chunk_*.pth`), and development artifacts, keeping container images lightweight and fast.
- **Root .gitignore:** Prevents leakage of credentials, tokens, local Windows paths, and uncompressed archives.
- **Public Demo Assets:** Bundles validated Scene 00955 (Gulf of Mexico) and Scene 00594 (False Positive Limitation Case) for instant demonstration.

---

## 4. Verification Test Summary

1. **Automated Unit & Integration Test Suite:**
   - Command: `python -m unittest discover tests`
   - Total Tests: **69 tests**
   - Result: **69 Passed, 0 Failures, 0 Errors (100% Pass Rate)**
2. **Dual-Scenario Demonstration Runner:**
   - Command: `python scripts/run_pre_deployment_demo.py`
   - Scenario 1 (Scene 00955): Inference (3.29s), Characterization ($2.273\text{ km}^2$), RK4 Hindcast ($\pm 8.0\text{ km}$), AIS correlation (2 candidates ranked, top score 0.97), Dashboard PNG & HTML report generated.
   - Scenario 2 (Scene 00594): Model FP detected, temporal bounds check confirms zero AIS overlap (2017 vs 2019/2026), outputs *"No sufficient AIS evidence"*, attributes **zero** candidate vessels.
3. **Model Checkpoint Immutability:**
   - Verified SHA-256 before and after tests: `cd648dafe044062a69760e71051f06830f62f7857bec42e5ea162400a20648d6` (100% match).

---

## 5. Rollback Procedure

If it is ever necessary to revert the GitHub repository to the legacy prototype:

```bash
# 1. Fetch all tags and branches from origin
git fetch origin

# 2. Checkout the permanent legacy branch
git checkout legacy/old-version

# 3. Reset main to the legacy commit SHA
git checkout main
git reset --hard legacy-pre-deployment

# 4. Force-push main back to the legacy state
git push origin main --force
```
*Note: The legacy version is permanently preserved in tag `legacy-pre-deployment` and branch `legacy/old-version` on GitHub and will never be lost.*

---

## 6. Public Hosting Instructions

### To run locally:
```bash
streamlit run app.py
```

### To run via Docker container:
```bash
docker build -t oceantrace:latest .
docker run --rm -p 8501:8501 oceantrace:latest
```

### Streamlit Community Cloud (Recommended Public Host):
1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Select repository: `chanderbala61-jpg/Oceantrace`.
3. Select branch: `main`.
4. Main file path: `app.py`.
5. Click **Deploy**. The platform automatically clones, installs `requirements.txt`, launches `app.py`, and provisions a public HTTPS URL (`https://<subdomain>.streamlit.app`).
