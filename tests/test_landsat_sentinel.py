from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from geodiff_gan.data.dataset import SentinelPatchDataset
from geodiff_gan.data.landsat_sentinel import (
    discover_landsat_products,
    landsat_acquisition_date,
    landsat_valid_mask,
    sentinel_acquisition_date,
)
from geodiff_gan.data.manifest import ManifestRecord, write_manifest
from geodiff_gan.losses import charbonnier
from geodiff_gan.metrics import psnr
from geodiff_gan.models.base import SwinIRBase


class LandsatSentinelDatasetTest(unittest.TestCase):
    def test_discovers_complete_products_in_flat_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            product = "LC09_L2SP_144044_20260527_20260528_02_T1"
            for layer in ("SR_B2", "SR_B3", "SR_B4", "QA_PIXEL", "QA_RADSAT"):
                (root / f"{product}_{layer}.TIF").touch()
            (root / f"{product}_MTL.txt").touch()
            products = discover_landsat_products(root)
            self.assertEqual(len(products), 1)
            self.assertEqual(products[0].product_id, product)
            self.assertIn("MTL.txt", products[0].files)

    def test_parses_sensor_acquisition_dates(self) -> None:
        self.assertEqual(
            landsat_acquisition_date(
                "LC09_L2SP_144044_20260527_20260528_02_T1"
            ).isoformat(),
            "2026-05-27",
        )
        self.assertEqual(
            sentinel_acquisition_date(
                "S2C_MSIL2A_20260527T050651_N0512_R019_T44RPQ_X.SAFE"
            ).isoformat(),
            "2026-05-27",
        )

    def test_landsat_qa_mask_rejects_cloud_and_saturation(self) -> None:
        qa = np.zeros((2, 2), dtype=np.uint16)
        qa[0, 1] = 1 << 3
        saturation = np.zeros_like(qa)
        saturation[1, 0] = 1
        valid = landsat_valid_mask(qa, saturation)
        np.testing.assert_array_equal(
            valid,
            np.array([[True, False], [False, True]]),
        )

    def test_paired_dataset_preserves_real_lr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patch = root / "pair.npz"
            hr = np.random.default_rng(1).random((3, 72, 72)).astype(np.float32)
            lr = np.random.default_rng(2).random((3, 24, 24)).astype(np.float32)
            np.savez_compressed(
                patch,
                hr=hr,
                lr=lr,
                clean_lr=lr,
                degradation=np.array([0.5, 1.0, 0.0, 0.0], dtype=np.float32),
                valid_mask_hr=np.ones((1, 72, 72), dtype=np.float32),
                valid_mask_lr=np.ones((1, 24, 24), dtype=np.float32),
            )
            manifest = root / "manifest.jsonl"
            write_manifest(
                manifest,
                [
                    ManifestRecord(
                        patch=str(patch),
                        tile_id="44RPQ",
                        split="train",
                        row=0,
                        col=0,
                        valid_fraction=1.0,
                        scale=3,
                    )
                ],
            )
            dataset = SentinelPatchDataset(
                manifest,
                split="train",
                scale=3,
                input_mode="paired",
                augment=False,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["lr"].shape), (3, 24, 24))
            self.assertEqual(tuple(sample["hr"].shape), (3, 72, 72))
            self.assertEqual(tuple(sample["valid_mask"].shape), (1, 72, 72))
            self.assertEqual(tuple(sample["valid_mask_lr"].shape), (1, 24, 24))
            self.assertTrue(np.allclose(sample["lr"].numpy(), lr))

    def test_resize_conv_base_supports_three_x(self) -> None:
        model = SwinIRBase(
            embed_dim=8,
            depth=1,
            heads=1,
            scale=3,
            upsample_mode="resize_conv",
        )
        output = model(torch.rand(1, 3, 16, 16))
        self.assertEqual(tuple(output.shape), (1, 3, 48, 48))

    def test_masked_metrics_ignore_invalid_target_pixels(self) -> None:
        prediction = torch.zeros(1, 3, 16, 16)
        target = prediction.clone()
        target[:, :, 0, 0] = 1
        mask = torch.ones(1, 1, 16, 16)
        mask[:, :, 0, 0] = 0
        self.assertEqual(
            float(charbonnier(prediction, target, epsilon=0.0, mask=mask)),
            0.0,
        )
        self.assertGreater(float(psnr(prediction, target, mask=mask)), 100.0)


if __name__ == "__main__":
    unittest.main()
