import os
import sys
import torch

print("Python executable:", sys.executable)
print("PyTorch version:", torch.__version__)

for name in ["latest.pth", "best.pth", "chunk_010.pth"]:
    p = os.path.join("outputs", "checkpoints", name)
    if os.path.exists(p):
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            chunk_num = ckpt.get("chunk_number")
            epoch = ckpt.get("epoch")
            scenes_start = ckpt.get("scenes_start")
            scenes_end = ckpt.get("scenes_end")
            scenes_proc = ckpt.get("scenes_processed")
            best_val = ckpt.get("best_val_iou")
            val_type = ckpt.get("validation_type")
            has_model = "model_state_dict" in ckpt
            has_opt = "optimizer_state_dict" in ckpt
            has_sched = "scheduler_state_dict" in ckpt
            # Check model parameter finiteness
            all_finite = True
            if has_model:
                for k, v in ckpt["model_state_dict"].items():
                    if not torch.isfinite(v).all():
                        all_finite = False
                        break
            print(f"[{name}] Chunk: {chunk_num}, Scenes: {scenes_start}-{scenes_end} (total {scenes_proc}), ValType: {val_type}, BestValIoU: {best_val}, Model: {has_model} (finite: {all_finite}), Opt: {has_opt}, Sched: {has_sched}")
        except Exception as e:
            print(f"[{name}] Error loading: {e}")
    else:
        print(f"[{name}] NOT FOUND at {p}")
