from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch import nn

from geodiff_gan.benchmark.models import (
    MODEL_SPECS,
    OFFICIAL_BUILDERS,
    ResizeConvHead,
    pixelshuffle_modules,
)
from geodiff_gan.benchmark.runner import (
    BenchmarkConfig,
    CroppedDataset,
    resolved_lr_crop,
)


class _ToyDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        lr = torch.arange(3 * 128 * 128, dtype=torch.float32).reshape(3, 128, 128)
        hr = torch.arange(3 * 512 * 512, dtype=torch.float32).reshape(3, 512, 512)
        return {
            "lr": lr,
            "clean_lr": lr.clone(),
            "hr": hr,
        }


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_defaults_to_official_architecture(self) -> None:
        config = BenchmarkConfig(
            model="swinir",
            source_root=Path("sources"),
            manifest=Path("manifest.jsonl"),
            output=Path("output"),
        )
        self.assertEqual(config.architecture_mode, "official")

    def test_resize_conv_head_is_x4_and_has_no_pixelshuffle(self) -> None:
        model = ResizeConvHead(32)
        output = model(torch.rand(1, 32, 16, 16))
        self.assertEqual(output.shape, (1, 3, 64, 64))
        self.assertEqual(pixelshuffle_modules(model), [])

    def test_pixelshuffle_modules_reports_offender(self) -> None:
        self.assertEqual(
            pixelshuffle_modules(nn.Sequential(nn.PixelShuffle(2))),
            ["0"],
        )

    def test_crop_preserves_lr_hr_alignment_shape(self) -> None:
        sample = CroppedDataset(_ToyDataset(), lr_crop=64)[0]
        self.assertEqual(sample["lr"].shape, (3, 64, 64))
        self.assertEqual(sample["clean_lr"].shape, (3, 64, 64))
        self.assertEqual(sample["hr"].shape, (3, 256, 256))

    def test_evaluation_crop_is_deterministic_and_centered(self) -> None:
        dataset = _ToyDataset()
        first = CroppedDataset(dataset, lr_crop=64, random_crop=False)[0]
        second = CroppedDataset(dataset, lr_crop=64, random_crop=False)[0]
        self.assertTrue(torch.equal(first["lr"], second["lr"]))
        self.assertTrue(torch.equal(first["hr"], second["hr"]))

    def test_native_size_is_model_specific(self) -> None:
        srformer = BenchmarkConfig(
            model="srformer",
            source_root=Path("sources"),
            manifest=Path("manifest.jsonl"),
            output=Path("output"),
        )
        swinir = BenchmarkConfig(
            model="swinir",
            source_root=Path("sources"),
            manifest=Path("manifest.jsonl"),
            output=Path("output"),
        )
        self.assertEqual(resolved_lr_crop(srformer), 48)
        self.assertEqual(resolved_lr_crop(swinir), 64)

    def test_conflicting_native_size_is_rejected(self) -> None:
        config = BenchmarkConfig(
            model="srformer",
            source_root=Path("sources"),
            manifest=Path("manifest.jsonl"),
            output=Path("output"),
            lr_crop=64,
        )
        with self.assertRaisesRegex(ValueError, "official 48x48"):
            resolved_lr_crop(config)

    def test_latest_official_models_are_registered(self) -> None:
        self.assertEqual(MODEL_SPECS["pft"].year, 2025)
        self.assertEqual(MODEL_SPECS["pft"].native_lr_size, 64)
        self.assertEqual(MODEL_SPECS["sat"].year, 2026)
        self.assertEqual(MODEL_SPECS["sat"].native_lr_size, 64)
        self.assertIn("pft", OFFICIAL_BUILDERS)
        self.assertIn("sat", OFFICIAL_BUILDERS)


if __name__ == "__main__":
    unittest.main()
