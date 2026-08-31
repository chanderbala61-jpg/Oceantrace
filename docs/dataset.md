# OceanTrace Dataset Inspection Report

## Overview
This document contains the detailed empirical inspection report for the satellite SAR Oil Spill Segmentation dataset located in `data/raw/`.

---

## 1. Dataset Directory Structure
```
data/raw/
├── Radar_data (1).rar (487.6 MB original archive)
├── train/
│   ├── images/                        (14 SAR satellite scenes in .tif format)
│   ├── masks/                         (14 segmentation target masks in .tif format)
│   ├── dataframe_train_dataset_256_90.csv  (21,744 training 256x256 patch annotations)
│   └── dataframe_val_dataset_256_90.csv    (7,249 validation 256x256 patch annotations)
└── test/
    ├── images/                        (7 SAR satellite scenes in .tif format)
    └── masks/                         (7 segmentation target masks in .tif format)
```

---

## 2. File & Scene Counts
* **Total Image Files**: **21** (14 in `train/images/`, 7 in `test/images/`)
* **Total Mask Files**: **21** (14 in `train/masks/`, 7 in `test/masks/`)
* **Total Image-Mask Pairs**: **21**

---

## 3. Formats & Dimensions

| Split | Scene Filename | Image Format | Mask Format | Dimensions (WxH) | Match |
|---|---|---|---|---|---|
| Train | `2018_08_21_.tif` | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 5701 x 4572 | Yes |
| Train | `2018_09_14_.tif` | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 3602 x 3216 | Yes |
| Train | `2018_12_07.tif`  | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 4424 x 2259 | Yes |
| Train | `2018_12_07_b.tif` | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 4922 x 2058 | Yes |
| Train | `2018_12_19.tif`  | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 5164 x 2421 | Yes |
| Train | `2018_12_19_b.tif` | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 4640 x 2165 | Yes |
| Train | `2018_12_31_b.tif` | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 5564 x 2965 | Yes |
| Train | `20190816.tif`     | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 3470 x 1942 | Yes |
| Train | `20190908.tif`     | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 3283 x 1891 | Yes |
| Train | `20200224.tif`     | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 3108 x 4195 | Yes |
| Train | `20200307.tif`     | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 3263 x 1994 | Yes |
| Train | `20200319.tif`     | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 2072 x 1476 | Yes |
| Train | `20200331.tif`     | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 3690 x 2007 | Yes |
| Train | `20200822.tif`     | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 2978 x 1839 | Yes |
| Test  | `2018_09_26.tif`   | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 5083 x 2555 | Yes |
| Test  | `2018_12_19_d.tif` | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 2340 x 1116 | Yes |
| Test  | `2018_12_19_e.tif` | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 3772 x 2380 | Yes |
| Test  | `2018_12_19_f_.tif`| TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 3250 x 2554 | Yes |
| Test  | `20191015.tif`     | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 5050 x 1683 | Yes |
| Test  | `20200224_b.tif`   | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 2590 x 1709 | Yes |
| Test  | `20200319b.tif`    | TIFF (`Format32bppArgb`) | TIFF (`Format32bppArgb`) | 2641 x 1528 | Yes |

---

## 4. Image-Mask Pairing & Target Encoding
* **Pairing Method**: Direct 1-to-1 filename match (`images/<filename>.tif` <-> `masks/<filename>.tif`).
* **Mask Classification Type**: **Binary** segmentation mask (`0.0` = background sea surface, `1.0` = oil spill region).

---

## 5. Patch Annotations & Class Balance (256x256 Patches, 90% Stride Overlap)

* **Training Set Patches (`dataframe_train_dataset_256_90.csv`)**:
  - **Total Patches**: 21,744
  - **Positive (Oil Spill, Class 1.0)**: 19,715 (90.67%)
  - **Negative (Background, Class 0.0)**: 2,029 (9.33%)

* **Validation Set Patches (`dataframe_val_dataset_256_90.csv`)**:
  - **Total Patches**: 7,249
  - **Positive (Oil Spill, Class 1.0)**: 6,563 (90.54%)
  - **Negative (Background, Class 0.0)**: 6,86 (9.46%)

* **Combined Total Patches**: **28,993** (26,278 positive, 2,715 negative)

---

## 6. Visual Verification Output
Visual verification side-by-side comparison images have been generated and saved under `outputs/visualizations/`:
* `outputs/visualizations/sample_pair_1.png`
* `outputs/visualizations/sample_pair_2.png`
* `outputs/visualizations/sample_pair_3.png`
* `outputs/visualizations/sample_pair_4.png`
* `outputs/visualizations/sample_pair_5.png`

---

## 7. Observations & Anomalies
* `test/images/20191015.tif`: Truncated Huffman table warning reported by archive extractor; however, the file opened successfully with full dimensions (`5050x1683`) and valid image header.
* High positive patch ratio in pre-extracted patch CSVs indicates sampling focused heavily on oil slick bounding regions.
