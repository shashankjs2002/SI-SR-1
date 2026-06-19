from __future__ import annotations

import unittest

import torch
from torch import nn

from geodiff_gan.benchmark.models import ResizeConvHead, assert_no_pixelshuffle
from geodiff_gan.benchmark.runner import CroppedDataset


class _ToyDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return {
            "lr": torch.rand(3, 128, 128),
            "clean_lr": torch.rand(3, 128, 128),
            "hr": torch.rand(3, 512, 512),
        }


class BenchmarkTests(unittest.TestCase):
    def test_resize_conv_head_is_x4_and_has_no_pixelshuffle(self) -> None:
        model = ResizeConvHead(32)
        output = model(torch.rand(1, 32, 16, 16))
        self.assertEqual(output.shape, (1, 3, 64, 64))
        assert_no_pixelshuffle(model)

    def test_pixelshuffle_assertion_rejects_offender(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_no_pixelshuffle(nn.Sequential(nn.PixelShuffle(2)))

    def test_crop_preserves_lr_hr_alignment_shape(self) -> None:
        sample = CroppedDataset(_ToyDataset(), lr_crop=64)[0]
        self.assertEqual(sample["lr"].shape, (3, 64, 64))
        self.assertEqual(sample["clean_lr"].shape, (3, 64, 64))
        self.assertEqual(sample["hr"].shape, (3, 256, 256))


if __name__ == "__main__":
    unittest.main()
