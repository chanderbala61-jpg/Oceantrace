"""
Google Colab GPU Training & Evaluation Entry Point
==================================================
Project: OceanTrace SegFormer Experiment 1
Task: Sentinel-1 SAR Oil Spill Semantic Segmentation

Usage on Colab:
  1. Smoke Test (verification before full run):
     python scripts/colab_train.py --smoke_test --data_dir /content/data/raw

  2. Full 30-Epoch Training:
     python scripts/colab_train.py --train --data_dir /content/data/raw

  3. Final Test Evaluation & Visualization:
     python scripts/colab_train.py --evaluate --data_dir /content/data/raw
"""

import os
import sys
import argparse
import torch

# Add workspace root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import run_training
from src.evaluate import run_evaluation
from scripts.pre_training_validation import run_pre_training_validation


def verify_colab_environment(require_gpu: bool = True):
    """
    Verifies that the environment has active CUDA GPU acceleration.
    """
    print("==================================================")
    print("COLAB RUNTIME HARDWARE VERIFICATION")
    print("==================================================")
    print(f"  - Python Executable: {sys.executable}")
    print(f"  - PyTorch Version:   {torch.__version__}")
    print(f"  - CUDA Available:    {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  - GPU Model:         {gpu_name}")
        print(f"  - Device Count:      {device_count}")
        print(f"  - Total VRAM:        {mem_gb:.2f} GB")
        print(f"  - CUDA Version:      {cuda_version}")
        print("  - Status:            [PASS] GPU ACCELERATION READY")
        print("==================================================")
        return True
    else:
        print("  - Status:            [FAIL] NO CUDA GPU DETECTED!")
        print("==================================================")
        if require_gpu:
            raise SystemExit(
                "\n[ERROR] No GPU available in this runtime! "
                "\nIn Google Colab, go to Runtime -> Change runtime type -> Hardware accelerator -> Select 'T4 GPU' or 'A100 GPU'. "
                "\nDo NOT train on CPU."
            )
        return False


def main():
    parser = argparse.ArgumentParser(description="Colab Entrypoint for OceanTrace SegFormer Exp 1")
    parser.add_argument("--config", type=str, default="configs/exp1_segformer_b0.yaml", help="Path to config YAML")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick 1-epoch smoke test on minimal samples")
    parser.add_argument("--validate", action="store_true", help="Run pre-training validation check on val scenes")
    parser.add_argument("--train", action="store_true", help="Run full 30-epoch training")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on the 7 official test scenes")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Checkpoint to evaluate or validate")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of val samples to visualize during --validate")
    parser.add_argument("--allow_cpu", action="store_true", help="Allow CPU (only for local debug)")
    args = parser.parse_args()

    # Verify GPU
    verify_colab_environment(require_gpu=not args.allow_cpu)

    if not args.smoke_test and not args.validate and not args.train and not args.evaluate:
        print("\nNo action selected. Please specify --smoke_test, --validate, --train, or --evaluate.")
        print("Example: python scripts/colab_train.py --smoke_test --data_dir /content/data/raw")
        return

    if args.smoke_test:
        print("\n>>> LAUNCHING SMOKE TEST PIPELINE...")
        res = run_training(
            config_path=args.config,
            raw_data_dir=args.data_dir,
            smoke_test=True,
            allow_cpu=args.allow_cpu
        )
        print("\n==================================================")
        print("[SMOKE TEST SUCCESSFUL]")
        print("All pipeline stages (Data -> Preprocessing -> SegFormer -> Forward -> Loss -> Backward -> Metrics -> Checkpoint) PASSED.")
        print("You can now safely proceed to full training upon approval.")
        print("==================================================")

    if args.validate:
        print("\n>>> LAUNCHING PRE-TRAINING VALIDATION CHECK...")
        result = run_pre_training_validation(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            raw_data_dir=args.data_dir,
            n_samples=args.n_samples,
            allow_cpu=args.allow_cpu
        )
        if not result["all_passed"]:
            raise SystemExit(
                "\n[ERROR] Pre-training validation FAILED. Fix the issues above before starting full training."
            )

    elif args.train:
        print("\n>>> LAUNCHING FULL 30-EPOCH TRAINING...")
        res = run_training(
            config_path=args.config,
            raw_data_dir=args.data_dir,
            smoke_test=False,
            allow_cpu=args.allow_cpu
        )
        print("\nTraining completed. Best Checkpoint:", res.get("best_checkpoint"))

    if args.evaluate:
        print("\n>>> LAUNCHING FINAL TEST EVALUATION...")
        run_evaluation(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            raw_data_dir=args.data_dir,
            allow_cpu=args.allow_cpu
        )


if __name__ == "__main__":
    main()
