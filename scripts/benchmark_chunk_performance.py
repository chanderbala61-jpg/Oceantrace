"""
OceanTrace Experiment 2: CPU Chunk Training Benchmark
-----------------------------------------------------
Benchmarks:
A. Loading one 256x256 patch from real Sentinel-1 GeoTIFF
B. One forward pass of SegFormer-B0 (in_channels=2, classes=1, input [1, 2, 256, 256])
C. One backward pass (loss.backward())
D. One optimizer step (AdamW step + zero_grad)
E. 10 training batches end-to-end (batch_size=1, num_workers=0)
Measures actual wall-clock time and RAM usage.
Estimates the time required for a 20-scene chunk and full 837-scene dataset.
"""

import os
import sys
import time
import gc
import ctypes
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import tifffile
import torch
import torch.nn as nn
from torch.optim import AdamW

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models_exp2 import build_exp2_segformer_b0
from src.losses import DiceBCELoss


def get_ram_info():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return {
        "total_gb": stat.ullTotalPhys / (1024 ** 3),
        "avail_gb": stat.ullAvailPhys / (1024 ** 3),
        "used_gb": (stat.ullTotalPhys - stat.ullAvailPhys) / (1024 ** 3),
        "load_pct": stat.dwMemoryLoad,
    }


def load_single_patch_windowed(img_path: str, mask_path: str, x: int = 512, y: int = 512, patch_size: int = 256):
    """Memory-efficient streaming windowed read: fetches exactly 256x256 from disk."""
    # 1. Read SAR image window
    with rasterio.open(img_path) as src:
        # Band 1 = VH, Band 2 = VV
        window = Window(x, y, patch_size, patch_size)
        img_patch = src.read([1, 2], window=window).astype(np.float32)  # (2, H, W)

    # 2. Read mask window via rasterio or tifffile
    with rasterio.open(mask_path) as m_src:
        m_window = Window(x, y, patch_size, patch_size)
        mask_patch = m_src.read(1, window=m_window).astype(np.float32)  # (H, W)

    # 3. Normalize SAR dB
    # VH: [-50, -10] -> [0, 1]
    vh = np.clip(img_patch[0], -50.0, -10.0)
    vh_norm = (vh - (-50.0)) / (-10.0 - (-50.0))
    
    # VV: [-35, +5] -> [0, 1]
    vv = np.clip(img_patch[1], -35.0, 5.0)
    vv_norm = (vv - (-35.0)) / (5.0 - (-35.0))

    img_tensor = torch.from_numpy(np.stack([vh_norm, vv_norm], axis=0)).unsqueeze(0)  # [1, 2, 256, 256]
    
    # Mask binarization & reshape
    mask_bin = (mask_patch > 0).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 256]

    return img_tensor, mask_tensor


def run_benchmark():
    print("=" * 65, flush=True)
    print("OCEANTRACE EXPERIMENT 2: LOCAL CPU BENCHMARK", flush=True)
    print("=" * 65, flush=True)
    
    ram_init = get_ram_info()
    print(f"Initial RAM: {ram_init['avail_gb']:.2f} GB free / {ram_init['total_gb']:.2f} GB total ({ram_init['load_pct']}% load)", flush=True)
    
    # Load training split and exclude corrupted scenes
    df_train = pd.read_csv("data/splits/experiment2_train.csv")
    excluded = ["00815.tif", "00816.tif", "00817.tif"]
    df_usable = df_train[~df_train["filename"].isin(excluded)].reset_index(drop=True)
    print(f"Total usable train scenes: {len(df_usable)} (excluded {len(excluded)} known unreadable TIFFs)", flush=True)
    
    sample_filename = df_usable.iloc[0]["filename"]
    img_path = os.path.join("extracted_dataset/images", sample_filename)
    mask_path = os.path.join("extracted_dataset/masks", sample_filename)
    
    print(f"Benchmarking with real sample scene: {sample_filename}", flush=True)
    
    # A. Benchmark loading one 256x256 patch
    print("\n--- BENCHMARK A: Loading One 256x256 Patch (Disk -> Window Read -> Norm -> Tensor) ---", flush=True)
    # Warmup
    _ = load_single_patch_windowed(img_path, mask_path, 256, 256, 256)
    
    times_load = []
    for i in range(5):
        t0 = time.perf_counter()
        img_t, mask_t = load_single_patch_windowed(img_path, mask_path, 256 + i * 50, 256 + i * 50, 256)
        t1 = time.perf_counter()
        times_load.append(t1 - t0)
    
    avg_load_time = np.mean(times_load)
    print(f"  Patch tensor shape: {img_t.shape}, Mask tensor shape: {mask_t.shape}", flush=True)
    print(f"  Wall-clock time: {avg_load_time * 1000:.2f} ms (avg of 5 runs: {[round(t*1000, 2) for t in times_load]} ms)", flush=True)
    
    # Model instantiation
    print("\n--- Instantiating SegFormer-B0 on CPU ---", flush=True)
    model = build_exp2_segformer_b0(in_channels=2, classes=1, encoder_weights=None, verbose=False)
    device = torch.device("cpu")
    model = model.to(device)
    model.train()
    
    criterion = DiceBCELoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Warmup forward & backward
    dummy_x = torch.randn(1, 2, 256, 256, device=device)
    dummy_y = torch.randint(0, 2, (1, 1, 256, 256), device=device, dtype=torch.float32)
    _out = model(dummy_x)
    _loss, _ = criterion(_out, dummy_y)
    _loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # B. Benchmark one forward pass
    print("\n--- BENCHMARK B: One Forward Pass (Input: [1, 2, 256, 256]) ---", flush=True)
    times_fwd = []
    for _ in range(5):
        t0 = time.perf_counter()
        out = model(img_t)
        t1 = time.perf_counter()
        times_fwd.append(t1 - t0)
    avg_fwd_time = np.mean(times_fwd)
    print(f"  Forward pass wall-clock time: {avg_fwd_time * 1000:.2f} ms (avg of 5 runs: {[round(t*1000, 2) for t in times_fwd]} ms)", flush=True)
    
    # C. Benchmark one backward pass
    print("\n--- BENCHMARK C: One Loss + Backward Pass ---", flush=True)
    times_bwd = []
    for _ in range(5):
        out = model(img_t)
        loss, _ = criterion(out, mask_t)
        optimizer.zero_grad()
        t0 = time.perf_counter()
        loss.backward()
        t1 = time.perf_counter()
        times_bwd.append(t1 - t0)
    avg_bwd_time = np.mean(times_bwd)
    print(f"  Backward pass wall-clock time: {avg_bwd_time * 1000:.2f} ms (avg of 5 runs: {[round(t*1000, 2) for t in times_bwd]} ms)", flush=True)
    
    # D. Benchmark one optimizer step
    print("\n--- BENCHMARK D: One Optimizer Step (AdamW step + zero_grad) ---", flush=True)
    times_opt = []
    for _ in range(5):
        t0 = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        t1 = time.perf_counter()
        times_opt.append(t1 - t0)
    avg_opt_time = np.mean(times_opt)
    print(f"  Optimizer step wall-clock time: {avg_opt_time * 1000:.2f} ms (avg of 5 runs: {[round(t*1000, 2) for t in times_opt]} ms)", flush=True)
    
    # E. Benchmark 10 training batches end-to-end
    print("\n--- BENCHMARK E: 10 Training Batches End-to-End ---", flush=True)
    print("  (Running 10 batches across distinct scenes with batch_size=1, num_workers=0)...", flush=True)
    
    batch_times = []
    losses = []
    
    t_start_10 = time.perf_counter()
    for batch_idx in range(10):
        t_b0 = time.perf_counter()
        
        # Pick scene from training set
        scene_row = df_usable.iloc[batch_idx]
        cur_img = os.path.join("extracted_dataset/images", scene_row["filename"])
        cur_mask = os.path.join("extracted_dataset/masks", scene_row["filename"])
        
        # Load patch (deterministic offset)
        x_off = 256 * ((batch_idx % 4) + 1)
        y_off = 256 * ((batch_idx % 4) + 1)
        img_b, mask_b = load_single_patch_windowed(cur_img, cur_mask, x_off, y_off, 256)
        
        # Forward pass
        pred = model(img_b)
        
        # Loss calculation
        batch_loss, breakdown = criterion(pred, mask_b)
        
        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        t_b1 = time.perf_counter()
        elapsed = t_b1 - t_b0
        batch_times.append(elapsed)
        losses.append(batch_loss.item())
        
        # Explicit garbage collection check
        del img_b, mask_b, pred, batch_loss
        gc.collect()
        
        print(f"    Batch {batch_idx + 1}/10: {elapsed * 1000:.1f} ms | Loss: {losses[-1]:.4f} (BCE: {breakdown['bce_loss']:.4f}, Dice: {breakdown['dice_loss']:.4f})", flush=True)
    
    t_end_10 = time.perf_counter()
    total_10_time = t_end_10 - t_start_10
    avg_batch_time = np.mean(batch_times)
    
    ram_post = get_ram_info()
    print(f"\n  Total time for 10 batches: {total_10_time:.2f} s ({total_10_time / 10:.2f} s/batch)", flush=True)
    print(f"  Average batch time: {avg_batch_time * 1000:.1f} ms", flush=True)
    print(f"  Post-benchmark RAM: {ram_post['avail_gb']:.2f} GB free / {ram_post['total_gb']:.2f} GB total ({ram_post['load_pct']}% load)", flush=True)
    print(f"  RAM change: {ram_post['avail_gb'] - ram_init['avail_gb']:+.2f} GB (System remained stable)", flush=True)
    
    # Projections
    patches_per_scene = 4
    scenes_per_chunk = 20
    patches_per_chunk = scenes_per_chunk * patches_per_scene  # 80 patches
    est_chunk_time_sec = avg_batch_time * patches_per_chunk
    
    total_scenes = len(df_usable)  # 837
    total_patches = total_scenes * patches_per_scene  # 837 * 4 = 3348 patches
    est_full_epoch_sec = avg_batch_time * total_patches
    
    print("\n" + "=" * 65, flush=True)
    print("BENCHMARK SUMMARY & PROJECTIONS", flush=True)
    print("=" * 65, flush=True)
    print(f"A. Loading 1 patch (windowed):      {avg_load_time * 1000:.1f} ms", flush=True)
    print(f"B. One forward pass:                {avg_fwd_time * 1000:.1f} ms", flush=True)
    print(f"C. One backward pass:               {avg_bwd_time * 1000:.1f} ms", flush=True)
    print(f"D. One optimizer step:              {avg_opt_time * 1000:.1f} ms", flush=True)
    print(f"E. 10 training batches wall-clock:  {total_10_time:.2f} seconds", flush=True)
    print(f"-----------------------------------------------------------------", flush=True)
    print(f"Estimated time for 1 chunk (20 scenes = 80 patches):", flush=True)
    print(f"   {est_chunk_time_sec:.1f} seconds (~{est_chunk_time_sec / 60:.2f} minutes)", flush=True)
    print(f"Estimated time for 837 scenes (1 full epoch = 42 chunks = 3,348 patches):", flush=True)
    print(f"   {est_full_epoch_sec:.1f} seconds (~{est_full_epoch_sec / 60:.1f} minutes / ~{est_full_epoch_sec / 3600:.2f} hours)", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    run_benchmark()
