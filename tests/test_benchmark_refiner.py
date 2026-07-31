from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from geodiff_gan.benchmark.models import pixelshuffle_modules
from geodiff_gan.benchmark.refiner import (
    FrozenBackboneRefiner,
    SensorNullspaceHighFrequencyRefiner,
)
from geodiff_gan.benchmark.refiner_runner import _paired_summary


class _ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            self.conv(x),
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )


class BenchmarkRefinerTests(unittest.TestCase):
    def _refiner(self) -> SensorNullspaceHighFrequencyRefiner:
        return SensorNullspaceHighFrequencyRefiner(
            channels=8,
            condition_dim=16,
            blocks_per_level=(1, 1, 1),
            max_residual=0.1,
            nullspace_iterations=1,
        )

    def test_refiner_starts_as_identity_and_preserves_x4_shape(self) -> None:
        refiner = self._refiner()
        lr = torch.rand(1, 3, 16, 16)
        base = F.interpolate(
            lr,
            scale_factor=4,
            mode="bicubic",
            align_corners=False,
        ).clamp(0, 1)
        degradation = torch.rand(1, 4)
        output = refiner(lr, base, degradation)
        self.assertEqual(output.refined.shape, (1, 3, 64, 64))
        self.assertEqual(output.residual.shape, (1, 3, 64, 64))
        self.assertEqual(output.confidence.shape, (1, 1, 64, 64))
        self.assertTrue(torch.equal(output.base, base))
        self.assertTrue(torch.allclose(output.refined, base, atol=1e-7))
        self.assertEqual(pixelshuffle_modules(refiner), [])

    def test_wrapper_freezes_backbone_and_keeps_it_in_eval_mode(self) -> None:
        pair = FrozenBackboneRefiner(_ToyBackbone(), self._refiner())
        pair.train()
        self.assertFalse(pair.backbone.training)
        self.assertTrue(pair.refiner.training)
        self.assertFalse(
            any(parameter.requires_grad for parameter in pair.backbone.parameters())
        )
        output = pair(torch.rand(1, 3, 16, 16), torch.rand(1, 4))
        output.refined.mean().backward()
        self.assertIsNotNone(pair.refiner.head[-1].weight.grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in pair.backbone.parameters())
        )

    def test_paired_summary_respects_metric_direction(self) -> None:
        rows = [
            {
                "base": {"psnr": 30.0, "l1": 0.03},
                "refined": {"psnr": 31.0, "l1": 0.02},
            },
            {
                "base": {"psnr": 29.0, "l1": 0.04},
                "refined": {"psnr": 29.5, "l1": 0.03},
            },
        ]
        summary = _paired_summary(rows, bootstrap_samples=20, seed=7)
        self.assertAlmostEqual(
            summary["paired"]["psnr"]["improved_patch_fraction"],
            1.0,
        )
        self.assertAlmostEqual(
            summary["paired"]["l1"]["improved_patch_fraction"],
            1.0,
        )
        self.assertGreater(
            summary["paired"]["psnr"]["mean_delta_refined_minus_base"],
            0,
        )
        self.assertLess(
            summary["paired"]["l1"]["mean_delta_refined_minus_base"],
            0,
        )

    def test_refiner_rejects_mismatched_base_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match x4"):
            self._refiner()(
                torch.rand(1, 3, 16, 16),
                torch.rand(1, 3, 60, 60),
                torch.rand(1, 4),
            )


if __name__ == "__main__":
    unittest.main()
