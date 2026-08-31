"""
OceanTrace - Automated Pipeline Runner

Orchestrates the full OceanTrace pipeline:
  1. Dataset validation
  2. Model training
  3. Evaluation
  4. Test prediction

Usage:
    python run_pipeline.py --epochs 3 --batch-size 8
"""

import os
import sys
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def main():
    parser = argparse.ArgumentParser(description='OceanTrace Full Pipeline Runner')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Raw data directory')
    parser.add_argument('--max-train-samples', type=int, default=None, help='Limit training samples')
    parser.add_argument('--max-val-samples', type=int, default=None, help='Limit validation samples')
    parser.add_argument('--skip-training', action='store_true', help='Skip training (use existing checkpoint)')
    args = parser.parse_args()

    print("=" * 60)
    print("         OCEANTRACE COMPLETE PIPELINE RUNNER")
    print("=" * 60)

    # Step 1: Training
    if not args.skip_training:
        print("\n[STEP 1/3] STARTING TRAINING...")
        from src.train import train_model
        checkpoint_path, history = train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            data_dir=args.data_dir,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples
        )
        print(f"Training complete. Best checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = 'models/checkpoints/best_unet_model.pth'
        print(f"\n[STEP 1/3] SKIPPING TRAINING. Using: {checkpoint_path}")

    # Step 2: Evaluation
    print("\n[STEP 2/3] RUNNING EVALUATION ON TEST SET...")
    from src.evaluate import evaluate_model
    metrics = evaluate_model(
        checkpoint_path=checkpoint_path,
        data_dir=args.data_dir
    )
    print(f"Test IoU: {metrics['mean_iou']:.4f} | Dice: {metrics['mean_dice']:.4f} | "
          f"Precision: {metrics['mean_precision']:.4f} | Recall: {metrics['mean_recall']:.4f}")

    # Step 3: Sample Prediction on first test image
    print("\n[STEP 3/3] RUNNING SAMPLE PREDICTION...")
    test_img_dir = os.path.join(args.data_dir, 'test', 'images')
    test_imgs = [f for f in os.listdir(test_img_dir) if f.endswith('.tif')] if os.path.exists(test_img_dir) else []

    if test_imgs:
        sample_img = os.path.join(test_img_dir, test_imgs[0])
        from src.predict import predict_image
        result = predict_image(
            image_path=sample_img,
            checkpoint_path=checkpoint_path
        )
        print(f"Prediction saved. Detected spill pixels: {result['detected_pixels']}")
    else:
        print("No test images found for sample prediction.")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE! Launch the dashboard with:")
    print("  streamlit run app/app.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
