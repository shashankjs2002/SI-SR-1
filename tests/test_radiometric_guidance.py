from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from geodiff_gan.data.dataset import SentinelPatchDataset
from geodiff_gan.data.manifest import ManifestRecord, write_manifest
from geodiff_gan.data.radiometry import fit_paired_radiometric_calibration
from geodiff_gan.models.system import GeoDiffGAN


class RadiometricGuidanceTest(unittest.TestCase):
    def test_rgb_base_uses_only_first_three_multispectral_channels(self) -> None:
        model = GeoDiffGAN(
            scale=3,
            input_channels=6,
            base_input_channels=3,
            output_channels=3,
            base_embed_dim=8,
            base_depth=1,
            base_heads=1,
            window_size=4,
            vae_channels=8,
            lr_channels=8,
            diffusion_widths=(8, 16, 24, 32),
            context_dim=16,
            mapper_channels=8,
            style_dim=8,
            decoder_channels=(8, 8, 8, 8),
            use_text_conditioning=False,
            use_degradation_conditioning=False,
        )
        lr = torch.rand(1, 6, 16, 16)
        first = model.predict_base(lr)
        modified = lr.clone()
        modified[:, 3:] = torch.rand_like(modified[:, 3:])
        second = model.predict_base(modified)
        self.assertEqual(tuple(first.shape), (1, 3, 48, 48))
        self.assertTrue(torch.equal(first, second))

    def test_dataset_calibrates_rgb_and_preserves_auxiliary_bands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lr_ms = np.full((6, 24, 24), 0.2, dtype=np.float32)
            hr = np.full((3, 72, 72), 0.4, dtype=np.float32)
            patch = root / "patch.npz"
            np.savez_compressed(
                patch,
                lr=lr_ms[:3],
                clean_lr=lr_ms[:3],
                lr_ms=lr_ms,
                clean_lr_ms=lr_ms,
                hr=hr,
                degradation=np.zeros(4, dtype=np.float32),
                valid_mask_hr=np.ones((1, 72, 72), dtype=np.float32),
                valid_mask_lr=np.ones((1, 24, 24), dtype=np.float32),
            )
            manifest = root / "manifest.jsonl"
            write_manifest(
                manifest,
                [ManifestRecord(str(patch), "tile", "train", 0, 0, 1.0)],
            )
            calibration = root / "calibration.json"
            calibration.write_text(
                json.dumps({"slope": [1.5, 1.0, 0.5], "offset": [0.1, 0.0, 0.05]}),
                encoding="utf-8",
            )
            sample = SentinelPatchDataset(
                manifest,
                split="train",
                scale=3,
                condition_key="lr_ms",
                radiometric_calibration=calibration,
                input_mode="paired",
                augment=False,
            )[0]
            expected = torch.tensor([0.4, 0.2, 0.15])[:, None, None]
            self.assertTrue(
                torch.allclose(sample["lr"][:3], expected.expand(-1, 24, 24))
            )
            self.assertTrue(
                torch.allclose(sample["lr"][3:], torch.full((3, 24, 24), 0.2))
            )
            self.assertTrue(
                torch.allclose(
                    sample["lr_raw_rgb"], torch.full((3, 24, 24), 0.2)
                )
            )

    def test_fit_uses_train_split_and_recovers_affine_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            slopes = np.array([1.1, 0.9, 1.2], dtype=np.float32)
            offsets = np.array([0.02, 0.03, -0.01], dtype=np.float32)
            rng = np.random.default_rng(7)
            for index, split in enumerate(("train", "train", "val")):
                lr = rng.uniform(0.05, 0.6, (3, 12, 12)).astype(np.float32)
                target_lr = lr * slopes[:, None, None] + offsets[:, None, None]
                hr = np.repeat(np.repeat(target_lr, 3, axis=1), 3, axis=2)
                patch = root / f"patch_{index}.npz"
                np.savez_compressed(
                    patch,
                    lr=lr,
                    clean_lr=lr,
                    hr=hr,
                    degradation=np.zeros(4, dtype=np.float32),
                    valid_mask_hr=np.ones((1, 36, 36), dtype=np.float32),
                    valid_mask_lr=np.ones((1, 12, 12), dtype=np.float32),
                )
                records.append(
                    ManifestRecord(str(patch), "tile", split, 0, index * 12, 1.0)
                )
            manifest = root / "manifest.jsonl"
            write_manifest(manifest, records)
            fitted = fit_paired_radiometric_calibration(
                manifest,
                condition_key="lr",
                scale=3,
                maximum_patches=10,
                pixels_per_patch=144,
                maximum_shift=1,
            )
            np.testing.assert_allclose(fitted["slope"], slopes, atol=1e-5)
            np.testing.assert_allclose(fitted["offset"], offsets, atol=1e-5)
            self.assertEqual(fitted["patches"], 2)
            self.assertTrue(
                all(
                    channel["calibrated_mae"] < channel["raw_mae"]
                    for channel in fitted["channels"]
                )
            )


if __name__ == "__main__":
    unittest.main()
