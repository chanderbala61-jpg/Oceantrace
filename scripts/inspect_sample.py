import sys
print("Starting inspect_sample.py", flush=True)

import numpy as np
import tifffile

for name in ['00000.tif', '00002.tif']:
    path = f"temp_audit_samples/images/{name}"
    print(f"Reading {path}...", flush=True)
    img = tifffile.imread(path)
    print(f"=== {name} ===", flush=True)
    print(f"Shape: {img.shape}", flush=True)
    print(f"Dtype: {img.dtype}", flush=True)
    print(f"Min: {np.min(img)}", flush=True)
    print(f"Max: {np.max(img)}", flush=True)
    print(f"Mean: {np.mean(img):.4f}", flush=True)
    print(f"Std: {np.std(img):.4f}", flush=True)
    if np.issubdtype(img.dtype, np.floating):
        print(f"NaN count: {np.isnan(img).sum()}", flush=True)
        print(f"Inf count: {np.isinf(img).sum()}", flush=True)
    else:
        print("NaN count: 0 (integer type)", flush=True)
        print("Inf count: 0 (integer type)", flush=True)

print("Done!", flush=True)
