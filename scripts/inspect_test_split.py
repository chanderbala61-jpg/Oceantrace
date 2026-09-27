import os
import sys
import hashlib
import pandas as pd
import rasterio

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_CSV = os.path.join(REPO_ROOT, "data", "splits", "experiment2_test.csv")
IMAGES_DIR = os.path.join(REPO_ROOT, "extracted_dataset", "images")
MASKS_DIR = os.path.join(REPO_ROOT, "extracted_dataset", "masks")
BEST_PTH = os.path.join(REPO_ROOT, "outputs", "checkpoints", "best.pth")

print("Checking test split:", TEST_CSV)
df_test = pd.read_csv(TEST_CSV)
print("Total rows in test split:", len(df_test))
print("Columns:", list(df_test.columns))

# Check best.pth hash
if os.path.exists(BEST_PTH):
    hasher = hashlib.sha256()
    with open(BEST_PTH, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    print("best.pth SHA256:", hasher.hexdigest())
else:
    print("best.pth NOT FOUND")

# Check test scenes readability
readable = []
unreadable = []
missing = []

for idx, row in df_test.iterrows():
    fn = row["filename"]
    img_p = os.path.join(IMAGES_DIR, fn)
    mask_p = os.path.join(MASKS_DIR, fn)
    
    if not os.path.exists(img_p) or not os.path.exists(mask_p):
        missing.append((fn, os.path.exists(img_p), os.path.exists(mask_p)))
        continue
        
    try:
        with rasterio.open(img_p) as src:
            _ = src.read(1, window=rasterio.windows.Window(0, 0, 256, 256))
        with rasterio.open(mask_p) as msrc:
            _ = msrc.read(1, window=rasterio.windows.Window(0, 0, 256, 256))
        readable.append(fn)
    except Exception as e:
        unreadable.append((fn, str(e)))

print(f"Readable: {len(readable)}")
print(f"Unreadable: {len(unreadable)}")
if unreadable:
    for u in unreadable:
        print("  Unreadable:", u)
print(f"Missing: {len(missing)}")
if missing:
    for m in missing:
        print("  Missing:", m)

# Also check 00813.tif specifically
p_00813_img = os.path.join(IMAGES_DIR, "00813.tif")
p_00813_mask = os.path.join(MASKS_DIR, "00813.tif")
print(f"00813.tif in df_test: {'00813.tif' in df_test['filename'].values}")
print(f"00813.tif image exists on disk: {os.path.exists(p_00813_img)}")
print(f"00813.tif mask exists on disk: {os.path.exists(p_00813_mask)}")
if os.path.exists(p_00813_img):
    try:
        with rasterio.open(p_00813_img) as s:
            _ = s.read(1)
        print("00813.tif is readable")
    except Exception as e:
        print("00813.tif read failed as expected:", e)
