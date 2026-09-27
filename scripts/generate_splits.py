"""
Generate Deterministic Scene-Level Splits for OceanTrace SegFormer Experiment 1
-------------------------------------------------------------------------------
Splits 14 training scenes into:
  - 10 Training Scenes
  - 4 Validation Scenes
And records the 7 official Test Scenes:
  - 7 Test Scenes (Strictly isolated for final evaluation)
"""

import os
import argparse
import pandas as pd
import numpy as np


TRAIN_RAW_SCENES = [
    "2018_08_21_.tif",
    "2018_09_14_.tif",
    "2018_12_07.tif",
    "2018_12_07_b.tif",
    "2018_12_19.tif",
    "2018_12_19_b.tif",
    "2018_12_31_b.tif",
    "20190816.tif",
    "20190908.tif",
    "20200224.tif",
    "20200307.tif",
    "20200319.tif",
    "20200331.tif",
    "20200822.tif",
]

TEST_RAW_SCENES = [
    "2018_09_26.tif",
    "2018_12_19_d.tif",
    "2018_12_19_e.tif",
    "2018_12_19_f_.tif",
    "20191015.tif",
    "20200224_b.tif",
    "20200319b.tif",
]


def generate_splits(output_dir: str, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort for deterministic permutation
    sorted_train_scenes = sorted(TRAIN_RAW_SCENES)
    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(len(sorted_train_scenes))
    
    # 10 train scenes, 4 val scenes
    train_indices = shuffled_indices[:10]
    val_indices = shuffled_indices[10:]
    
    train_scenes = [sorted_train_scenes[i] for i in sorted(train_indices)]
    val_scenes = [sorted_train_scenes[i] for i in sorted(val_indices)]
    test_scenes = sorted(TEST_RAW_SCENES)
    
    # Create DataFrames
    df_train = pd.DataFrame({
        "scene_name": train_scenes,
        "split": "train",
        "relative_img_path": [os.path.join("train", "images", s).replace("\\", "/") for s in train_scenes],
        "relative_mask_path": [os.path.join("train", "masks", s).replace("\\", "/") for s in train_scenes],
    })
    
    df_val = pd.DataFrame({
        "scene_name": val_scenes,
        "split": "val",
        "relative_img_path": [os.path.join("train", "images", s).replace("\\", "/") for s in val_scenes],
        "relative_mask_path": [os.path.join("train", "masks", s).replace("\\", "/") for s in val_scenes],
    })
    
    df_test = pd.DataFrame({
        "scene_name": test_scenes,
        "split": "test",
        "relative_img_path": [os.path.join("test", "images", s).replace("\\", "/") for s in test_scenes],
        "relative_mask_path": [os.path.join("test", "masks", s).replace("\\", "/") for s in test_scenes],
    })
    
    train_path = os.path.join(output_dir, "train_scenes.csv")
    val_path = os.path.join(output_dir, "val_scenes.csv")
    test_path = os.path.join(output_dir, "test_scenes.csv")
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print("==================================================")
    print("SCENE-LEVEL SPLIT SUMMARY (Seed = {})".format(seed))
    print("==================================================")
    print(f"Train scenes ({len(train_scenes)}):")
    for s in train_scenes:
        print(f"  - {s}")
    print(f"\nValidation scenes ({len(val_scenes)}):")
    for s in val_scenes:
        print(f"  - {s}")
    print(f"\nTest scenes ({len(test_scenes)}):")
    for s in test_scenes:
        print(f"  - {s}")
    print("\nSaved split files to:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")
    print("==================================================")
    
    return df_train, df_val, df_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate deterministic scene splits")
    parser.add_argument("--output_dir", type=str, default="data/splits", help="Directory to save split CSVs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    generate_splits(args.output_dir, args.seed)
