from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from geodiff_gan.data.dataset import SentinelPatchDataset
from geodiff_gan.data.diverse import (
    annotate_scene_classes,
    export_portable_dataset,
    infer_scene_class,
    validate_category_splits,
)
from geodiff_gan.data.landsat_sentinel import (
    discover_landsat_products,
    landsat_acquisition_date,
    landsat_valid_mask,
    sentinel_acquisition_date,
)
from geodiff_gan.data.manifest import (
    ManifestRecord,
    assign_within_tile_spatial_splits,
    build_within_tile_spatial_folds,
    load_manifest,
    validate_within_tile_spatial_isolation,
    write_manifest,
)
from geodiff_gan.losses import charbonnier
from geodiff_gan.metrics import psnr
from geodiff_gan.models.base import SwinIRBase


class LandsatSentinelDatasetTest(unittest.TestCase):
    def test_discovers_complete_products_in_flat_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            product = "LC09_L2SP_144044_20260527_20260528_02_T1"
            for layer in (
                "SR_B2",
                "SR_B3",
                "SR_B4",
                "SR_B5",
                "SR_B6",
                "SR_B7",
                "QA_PIXEL",
                "QA_RADSAT",
            ):
                (root / f"{product}_{layer}.TIF").touch()
            (root / f"{product}_MTL.txt").touch()
            products = discover_landsat_products(root)
            self.assertEqual(len(products), 1)
            self.assertEqual(products[0].product_id, product)
            self.assertIn("MTL.txt", products[0].files)
            self.assertTrue(
                all(
                    layer in products[0].files
                    for layer in ("SR_B5", "SR_B6", "SR_B7")
                )
            )

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

    def test_diverse_metadata_and_portable_relative_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "pair.npz"
            source.parent.mkdir()
            np.savez_compressed(
                source,
                lr=np.zeros((3, 8, 8), dtype=np.float32),
                hr=np.zeros((3, 24, 24), dtype=np.float32),
            )
            records = [
                ManifestRecord(
                    patch=str(source), tile_id="44RPQ", split=split,
                    row=index * 24, col=0, valid_fraction=1.0, scale=3,
                )
                for index, split in enumerate(("train", "val", "test"))
            ]
            records = annotate_scene_classes(
                records, category_by_tile={"44RPQ": "built-up"}
            )
            self.assertTrue(all(record.scene_class == "urban" for record in records))
            rows = validate_category_splits(records, minimum_test_fraction=0.1)
            self.assertEqual(rows[0]["test"], 1)
            manifest = export_portable_dataset(records, root / "portable")
            text_rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertFalse(Path(text_rows[0]["patch"]).is_absolute())
            loaded = load_manifest(manifest, resolve_paths=True)
            self.assertTrue(all(Path(record.patch).is_file() for record in loaded))

    def test_scene_class_path_inference_is_conservative(self) -> None:
        self.assertEqual(infer_scene_class(Path("dataset") / "urban" / "tile"), "urban")
        self.assertEqual(infer_scene_class(Path("dataset") / "cropland" / "tile"), "agriculture")
        self.assertIsNone(infer_scene_class(Path("dataset") / "unknown" / "tile"))
        with self.assertRaises(ValueError):
            infer_scene_class(Path("dataset") / "urban" / "forest" / "tile")

    def test_paired_dataset_loads_real_multispectral_lr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patch = root / "pair_ms.npz"
            rng = np.random.default_rng(22)
            hr = rng.random((3, 72, 72)).astype(np.float32)
            lr = rng.random((3, 24, 24)).astype(np.float32)
            lr_ms = rng.random((6, 24, 24)).astype(np.float32)
            lr_ms[:3] = lr
            np.savez_compressed(
                patch,
                hr=hr,
                lr=lr,
                clean_lr=lr,
                lr_ms=lr_ms,
                clean_lr_ms=lr_ms,
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
            sample = SentinelPatchDataset(
                manifest,
                split="train",
                scale=3,
                condition_key="lr_ms",
                input_mode="paired",
                output_channels=3,
                augment=False,
            )[0]
            self.assertEqual(tuple(sample["lr"].shape), (6, 24, 24))
            self.assertEqual(tuple(sample["lr_rgb"].shape), (3, 24, 24))
            self.assertTrue(np.allclose(sample["lr"].numpy(), lr_ms))
            self.assertTrue(np.allclose(sample["lr_rgb"].numpy(), lr))

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

    def test_within_tile_spatial_split_has_guard_bands(self) -> None:
        records = [
            ManifestRecord(
                patch=f"patch_{row}_{col}.npz",
                tile_id="44QLL",
                split="train",
                row=row,
                col=col,
                valid_fraction=1.0,
                scale=3,
            )
            for row in (0, 80)
            for col in range(0, 800, 80)
        ]
        report = assign_within_tile_spatial_splits(
            records,
            patch_size=100,
            train_fraction=0.8,
            validation_fraction=0.1,
            minimum_validation_fraction=0.05,
            minimum_test_fraction=0.1,
        )
        counts = {split: 0 for split in ("train", "val", "test", "discard")}
        for record in records:
            counts[record.split] += 1
        self.assertGreater(counts["train"], 0)
        self.assertGreater(counts["val"], 0)
        self.assertGreater(counts["test"], 0)
        self.assertGreater(counts["discard"], 0)
        retained = counts["train"] + counts["val"] + counts["test"]
        self.assertGreaterEqual(counts["test"] / retained, 0.1)
        self.assertEqual(report["44QLL"]["axis"], "col")
        validate_within_tile_spatial_isolation(records, patch_size=100)

        by_split = {
            split: [record for record in records if record.split == split]
            for split in ("train", "val", "test")
        }
        self.assertLessEqual(
            max(record.col + 100 for record in by_split["train"]),
            min(record.col for record in by_split["val"]),
        )
        self.assertLessEqual(
            max(record.col + 100 for record in by_split["val"]),
            min(record.col for record in by_split["test"]),
        )

    def test_spatial_kfold_populates_every_tile_without_overlap(self) -> None:
        records = [
            ManifestRecord(
                patch=f"{tile}_patch_{row}_{col}.npz",
                tile_id=tile,
                split="train",
                row=row,
                col=col,
                valid_fraction=1.0,
                scale=3,
            )
            for tile in ("44QLL", "44QMM")
            for row in (0, 80)
            for col in range(0, 4000, 80)
        ]
        manifests, reports = build_within_tile_spatial_folds(
            records,
            patch_size=100,
            folds=5,
        )
        self.assertEqual(len(manifests), 5)
        self.assertEqual(set(reports), {"44QLL", "44QMM"})
        for fold_records in manifests:
            for tile in ("44QLL", "44QMM"):
                tile_records = [
                    record for record in fold_records if record.tile_id == tile
                ]
                counts = Counter(record.split for record in tile_records)
                self.assertGreater(counts["train"], 0)
                self.assertGreater(counts["val"], 0)
                self.assertGreater(counts["test"], 0)
                active = [
                    record for record in tile_records if record.split != "discard"
                ]
                for left_index, left in enumerate(active):
                    for right in active[left_index + 1 :]:
                        if left.split == right.split:
                            continue
                        row_overlap = not (
                            left.row + 100 <= right.row
                            or right.row + 100 <= left.row
                        )
                        col_overlap = not (
                            left.col + 100 <= right.col
                            or right.col + 100 <= left.col
                        )
                        self.assertFalse(row_overlap and col_overlap)


if __name__ == "__main__":
    unittest.main()
