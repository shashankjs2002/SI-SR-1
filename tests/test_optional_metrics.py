import unittest

import torch

from geodiff_gan.metrics import (
    ergas,
    remote_sensing_metrics,
    spatial_correlation_coefficient,
    spectral_angle_mapper,
    uiqi,
)


class RemoteSensingMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(7)
        self.target = torch.rand(2, 3, 24, 24, generator=generator) * 0.8 + 0.1

    def test_identical_images_have_ideal_scores(self) -> None:
        values = remote_sensing_metrics(self.target, self.target, scale=3)
        self.assertAlmostEqual(values["ergas"], 0.0, places=6)
        self.assertAlmostEqual(values["sam_degrees"], 0.0, places=3)
        self.assertAlmostEqual(values["uiqi"], 1.0, places=6)
        self.assertAlmostEqual(values["scc"], 1.0, places=6)

    def test_metrics_detect_radiometric_and_spatial_distortion(self) -> None:
        prediction = (self.target * 0.65 + 0.1).roll(shifts=1, dims=-1)
        self.assertGreater(float(ergas(prediction, self.target, scale=3)), 0.0)
        self.assertGreater(float(spectral_angle_mapper(prediction, self.target)), 0.0)
        self.assertLess(float(uiqi(prediction, self.target)), 1.0)
        self.assertLess(
            float(spatial_correlation_coefficient(prediction, self.target)),
            1.0,
        )

    def test_mask_excludes_invalid_region(self) -> None:
        prediction = self.target.clone()
        prediction[:, :, :4, :4] = 1.0 - prediction[:, :, :4, :4]
        mask = torch.ones(2, 1, 24, 24)
        mask[:, :, :4, :4] = 0.0
        values = remote_sensing_metrics(prediction, self.target, scale=3, mask=mask)
        self.assertAlmostEqual(values["ergas"], 0.0, places=6)
        self.assertAlmostEqual(values["sam_degrees"], 0.0, places=3)
        self.assertAlmostEqual(values["uiqi"], 1.0, places=6)
        self.assertAlmostEqual(values["scc"], 1.0, places=6)

    def test_ergas_uses_resolution_ratio(self) -> None:
        prediction = self.target + 0.05
        score_3x = float(ergas(prediction, self.target, scale=3))
        score_6x = float(ergas(prediction, self.target, scale=6))
        self.assertAlmostEqual(score_3x, score_6x * 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
