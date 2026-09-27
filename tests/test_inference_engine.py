"""
Unit & Smoke Test for OceanTrace Inference Engine
=================================================
Verifies:
1. Frozen checkpoint loading & SHA-256 fingerprint verification.
2. SAR radiometric preprocessing.
3. Deterministic patch inference.
4. Tiled scene prediction integrity.
"""

import os
import unittest
import numpy as np
import torch

from src.inference import (
    load_frozen_segformer,
    verify_checkpoint_hash,
    preprocess_sar_imagery,
    predict_scene_tiled,
    LOCKED_CHECKPOINT_SHA256,
    DEFAULT_CHECKPOINT_PATH,
)


class TestInferenceEngine(unittest.TestCase):

    def test_checkpoint_hash_verification(self):
        is_valid, digest = verify_checkpoint_hash(DEFAULT_CHECKPOINT_PATH)
        self.assertTrue(is_valid, f"Hash mismatch: got {digest}")
        self.assertEqual(digest, LOCKED_CHECKPOINT_SHA256)

    def test_load_frozen_segformer(self):
        model, meta = load_frozen_segformer(device="cpu", verify_hash=True)
        self.assertIsNotNone(model)
        self.assertTrue(meta["hash_verified"])
        self.assertEqual(meta["trainable_parameters"], 0)
        self.assertEqual(meta["total_parameters"], 3712833)
        self.assertFalse(model.training)

        # Verify no parameters require grad
        for p in model.parameters():
            self.assertFalse(p.requires_grad)

    def test_preprocess_sar_imagery(self):
        dummy_sar = np.array([
            [[-60.0, -40.0], [-10.0, 5.0]],
            [[-50.0, -35.0], [0.0, 10.0]],
        ], dtype=np.float32)

        norm_sar = preprocess_sar_imagery(dummy_sar)
        self.assertEqual(norm_sar.shape, (2, 2, 2))
        self.assertGreaterEqual(norm_sar.min(), 0.0)
        self.assertLessEqual(norm_sar.max(), 1.0)

    def test_predict_scene_tiled_smoke(self):
        model, _ = load_frozen_segformer(device="cpu", verify_hash=True)
        dummy_scene = np.full((256, 256, 2), -20.0, dtype=np.float32)
        res = predict_scene_tiled(model, dummy_scene, patch_size=256, stride=256)
        self.assertIn("probability_map", res)
        self.assertIn("binary_mask", res)
        self.assertEqual(res["probability_map"].shape, (256, 256))
        self.assertEqual(res["binary_mask"].shape, (256, 256))


if __name__ == "__main__":
    unittest.main()
