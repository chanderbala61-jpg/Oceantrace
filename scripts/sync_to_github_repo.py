"""
OceanTrace GitHub Repository Synchronization Script
===================================================
Prepares temp_github_repo with the exact validated deployment version,
removing obsolete prototype files while preserving Git history.
"""

import os
import shutil
import glob

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEST_ROOT = os.path.join(SRC_ROOT, "temp_github_repo")

# 1. Clean obsolete directories and files from temp_github_repo
obsolete_items = [
    os.path.join(DEST_ROOT, "app"),
    os.path.join(DEST_ROOT, "docs"),
    os.path.join(DEST_ROOT, "models"),
    os.path.join(DEST_ROOT, "run_pipeline.py"),
    os.path.join(DEST_ROOT, "test_pipeline.py"),
    os.path.join(DEST_ROOT, "packages.txt"),
]

for item in obsolete_items:
    if os.path.isdir(item):
        shutil.rmtree(item)
        print(f"[REMOVED OBSOLETE DIR] {os.path.basename(item)}")
    elif os.path.isfile(item):
        os.remove(item)
        print(f"[REMOVED OBSOLETE FILE] {os.path.basename(item)}")

# 2. Files to copy directly to root
root_files = [
    "app.py",
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
    "requirements.txt",
    "README.md",
    "config.yaml",
    ".env.example",
]

for rf in root_files:
    src_p = os.path.join(SRC_ROOT, rf)
    dest_p = os.path.join(DEST_ROOT, rf)
    if os.path.isfile(src_p):
        shutil.copy2(src_p, dest_p)
        print(f"[COPIED ROOT FILE] {rf}")

# 3. Directories to sync completely (ignoring __pycache__)
sync_dirs = [
    "src",
    "tests",
    "scripts",
]

def ignore_pycache(dirpath, contents):
    return [c for c in contents if c in ("__pycache__", ".pytest_cache") or c.endswith(".pyc")]

for sd in sync_dirs:
    src_p = os.path.join(SRC_ROOT, sd)
    dest_p = os.path.join(DEST_ROOT, sd)
    if os.path.exists(dest_p):
        shutil.rmtree(dest_p)
    shutil.copytree(src_p, dest_p, ignore=ignore_pycache)
    print(f"[SYNCED DIRECTORY] {sd}")

# 4. Outputs to bundle (locked checkpoints, locked test evaluation, audit, investigations)
outputs_to_bundle = [
    ("outputs/checkpoints/best.pth", "outputs/checkpoints/best.pth"),
    ("outputs/experiment2_test_evaluation", "outputs/experiment2_test_evaluation"),
    ("outputs/ais_false_detection_link", "outputs/ais_false_detection_link"),
    ("outputs/investigations", "outputs/investigations"),
    ("outputs/pre_deployment_audit", "outputs/pre_deployment_audit"),
    ("outputs/deployment", "outputs/deployment"),
]

for rel_src, rel_dest in outputs_to_bundle:
    src_p = os.path.join(SRC_ROOT, rel_src)
    dest_p = os.path.join(DEST_ROOT, rel_dest)
    os.makedirs(os.path.dirname(dest_p), exist_ok=True)
    if os.path.isfile(src_p):
        shutil.copy2(src_p, dest_p)
        print(f"[COPIED OUTPUT FILE] {rel_dest}")
    elif os.path.isdir(src_p):
        if os.path.exists(dest_p):
            shutil.rmtree(dest_p)
        shutil.copytree(src_p, dest_p, ignore=ignore_pycache)
        print(f"[SYNCED OUTPUT DIR] {rel_dest}")

# 5. Lightweight demo assets
demo_assets = [
    ("extracted_dataset/images/00955.tif", "extracted_dataset/images/00955.tif"),
    ("extracted_dataset/images/00594.tif", "extracted_dataset/images/00594.tif"),
    ("extracted_dataset/masks/00955.tif", "extracted_dataset/masks/00955.tif"),
    ("extracted_dataset/masks/00594.tif", "extracted_dataset/masks/00594.tif"),
    ("see thsis/Ocean Trace 2/data/synthetic_ais_demo.csv", "see thsis/Ocean Trace 2/data/synthetic_ais_demo.csv"),
    ("data/splits/experiment2_test.csv", "data/splits/experiment2_test.csv"),
]

for rel_src, rel_dest in demo_assets:
    src_p = os.path.join(SRC_ROOT, rel_src)
    dest_p = os.path.join(DEST_ROOT, rel_dest)
    if os.path.exists(src_p):
        os.makedirs(os.path.dirname(dest_p), exist_ok=True)
        shutil.copy2(src_p, dest_p)
        print(f"[COPIED DEMO ASSET] {rel_dest}")

print("\n[SUCCESS] Synchronization to temp_github_repo completed cleanly.")
