"""
Unit Tests for OceanTrace SAR Look-Alike & Ambiguity Assessment Module
---------------------------------------------------------------------
Verifies:
  1. Empty mask handling
  2. Low-wind calm sea ambiguity detection
  3. High-wind dispersion flag
  4. Highly linear structure / wake elongation detection
  5. Radiometric contrast calculation between slick and ocean background
  6. Optimal contrast and wind regime yielding reliable classification
  7. Strict adherence to non-fabricated explainable criteria
"""

import unittest
import numpy as np

from src.look_alike import LookAlikeAnalyzer, LookAlikeAssessment


class TestLookAlikeAssessment(unittest.TestCase):

    def setUp(self):
        self.analyzer = LookAlikeAnalyzer(
            min_contrast_db=3.5,
            optimal_contrast_db=7.0,
            low_wind_threshold_ms=3.0,
            high_wind_threshold_ms=12.0
        )

    # 1. Empty mask handling
    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        res = self.analyzer.assess(spill_id="EMPTY_01", mask=mask)
        self.assertEqual(res.ambiguity_level, "HIGH")
        self.assertEqual(res.ambiguity_score, 1.0)
        self.assertIn("NO_ACTIVE_SLICK", res.flags)

    # 2. Low wind ambiguity flag
    def test_low_wind_flag(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 1
        res = self.analyzer.assess(
            spill_id="LOW_WIND_01",
            mask=mask,
            wind_speed_ms=1.8  # Calm water look-alike condition
        )
        self.assertIn("LOW_WIND_AMBIGUITY_ZONE", res.flags)
        self.assertGreaterEqual(res.ambiguity_score, 0.40)
        self.assertTrue(any("calm" in w.lower() or "attenuated" in w.lower() for w in res.warnings))

    # 3. High wind dispersion flag
    def test_high_wind_flag(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 1
        res = self.analyzer.assess(
            spill_id="HIGH_WIND_01",
            mask=mask,
            wind_speed_ms=15.0  # Rough sea dispersion condition
        )
        self.assertIn("HIGH_WIND_DISPERSION_ZONE", res.flags)
        self.assertTrue(any("dissipation" in w.lower() or "dispersion" in w.lower() for w in res.warnings))

    # 4. High elongation linear feature (wake / internal wave)
    def test_high_elongation(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Narrow linear feature: width=4, length=120 (aspect ratio ~30)
        mask[98:102, 40:160] = 1
        res = self.analyzer.assess(
            spill_id="WAKE_01",
            mask=mask,
            wind_speed_ms=6.0
        )
        self.assertIn("HIGH_ELONGATION_LINEAR_STRUCTURE", res.flags)
        self.assertGreater(res.aspect_ratio, 10.0)

    # 5. Radiometric contrast calculation
    def test_radiometric_contrast(self):
        # Create synthetic SAR image in dB:
        # Sea background ~ -15 dB
        # Dark slick area ~ -25 dB (10 dB contrast)
        h, w = 100, 100
        sar_img = np.full((h, w), -15.0, dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[40:60, 40:60] = 1
        sar_img[mask == 1] = -25.0

        res = self.analyzer.assess(
            spill_id="CONTRAST_TEST_01",
            mask=mask,
            sar_image=sar_img,
            wind_speed_ms=7.0
        )
        self.assertIsNotNone(res.contrast_db)
        self.assertAlmostEqual(res.contrast_db, 10.0, delta=1.0)
        self.assertEqual(res.ambiguity_level, "LOW")
        self.assertTrue(res.is_reliable_detection)

    # 6. Low contrast ambiguity
    def test_low_contrast_ambiguity(self):
        h, w = 100, 100
        sar_img = np.full((h, w), -18.0, dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[40:60, 40:60] = 1
        # Weak damping of only 1.5 dB
        sar_img[mask == 1] = -19.5

        res = self.analyzer.assess(
            spill_id="LOW_CONTRAST_01",
            mask=mask,
            sar_image=sar_img,
            wind_speed_ms=6.0
        )
        self.assertIn("LOW_RADIOMETRIC_CONTRAST", res.flags)
        self.assertGreater(res.ambiguity_score, 0.35)


if __name__ == "__main__":
    unittest.main()
