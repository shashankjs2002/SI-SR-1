from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from geodiff_gan.config import load_config
from geodiff_gan.cli.caption_qwen import _resolve_model_class
from geodiff_gan.data import SentinelPatchDataset
from geodiff_gan.data.manifest import (
    ManifestRecord,
    deterministic_split,
    load_manifest,
    validate_tile_split_isolation,
    write_manifest,
)
from geodiff_gan.data.sentinel import (
    reassign_product_splits,
    split_for_product,
)
from geodiff_gan.diagnostics import DiagnosticRecorder, tensor_statistics
from geodiff_gan.metrics import edge_f1
from geodiff_gan.models.base import WindowTransformerBlock
from geodiff_gan.models.blocks import CrossAttention2d, high_pass
from geodiff_gan.models.degradation import back_project, random_degradation, sensor_degrade
from geodiff_gan.models.system import GeoDiffGAN
from geodiff_gan.text import HashTextEncoder, PromptBatch, augment_prompts


ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs/smoke.yaml", ROOT / "configs/default.yaml")

    def test_model_shapes_and_sampling_modes(self) -> None:
        torch.manual_seed(1)
        model = GeoDiffGAN.from_config(self.config).eval()
        lr = torch.rand(1, 3, 16, 16)
        degradation = torch.rand(1, 4)
        context = HashTextEncoder(32, 8)(["dense urban roofs"])
        base = model.base(lr)
        self.assertEqual(base.shape, (1, 3, 64, 64))
        residual = torch.rand_like(base) - 0.5
        latent, mean, log_variance = model.vae.encode(residual, sample=False)
        self.assertEqual(latent.shape, (1, 4, 8, 8))
        self.assertEqual(model.vae.downsample_factor, 8)
        self.assertEqual(mean.shape, log_variance.shape)
        batch = model.prepare_diffusion_batch(latent)
        features = model.lr_encoder(lr)
        velocity = model.predict_velocity(
            batch.noisy,
            batch.timesteps,
            context,
            degradation,
            "sr",
            features,
        )
        self.assertEqual(velocity.shape, latent.shape)
        output = model.decode_latent(
            latent, lr, context, degradation, mode="sr", base=base, back_projection_steps=1
        )
        self.assertEqual(output.image.shape, base.shape)
        self.assertEqual(output.evidence_confidence.shape, (1, 1, 8, 8))
        self.assertEqual(output.edit_permission.shape, (1, 1, 8, 8))
        self.assertTrue(
            torch.equal(
                output.edit_permission,
                torch.zeros_like(output.edit_permission),
            )
        )
        self.assertEqual(output.raw_detail_residual.shape, base.shape)
        self.assertEqual(output.raw_edit_residual.shape, base.shape)
        self.assertEqual(output.abstention_map.shape, (1, 1, 64, 64))
        self.assertFalse(output.metadata[0]["synthetic_edit"])
        edit = model.decode_latent(
            latent, lr, context, degradation, mode="edit", base=base, back_projection_steps=0
        )
        self.assertTrue(edit.metadata[0]["synthetic_edit"])
        self.assertGreater(float(edit.edit_permission.detach().mean()), 0.0)

    def test_uncertainty_abstention_returns_to_base(self) -> None:
        model = GeoDiffGAN.from_config(self.config).eval()
        image = torch.ones(1, 3, 32, 32)
        base = torch.zeros_like(image)
        evidence = torch.ones(1, 1, 4, 4)
        uncertainty = torch.full((1, 32, 32), 1.0)
        abstained, confidence, abstention = model.apply_uncertainty_abstention(
            image, base, evidence, uncertainty
        )
        self.assertLess(float(abstained.abs().max()), 1e-5)
        self.assertLess(float(confidence.max()), 1e-5)
        self.assertGreater(float(abstention.min()), 0.99999)

    def test_prompt_augmentation_reports_policy_labels(self) -> None:
        augmented = augment_prompts(
            ["coastal settlement"],
            null_probability=1.0,
            paraphrase_probability=0.0,
            mismatch_probability=0.0,
            return_metadata=True,
        )
        self.assertIsInstance(augmented, PromptBatch)
        assert isinstance(augmented, PromptBatch)
        self.assertEqual(augmented.prompts, [""])
        self.assertEqual(augmented.kinds, ["null"])
        mismatch = augment_prompts(
            ["coastal settlement"],
            null_probability=0.0,
            paraphrase_probability=0.0,
            mismatch_probability=1.0,
            return_metadata=True,
        )
        assert isinstance(mismatch, PromptBatch)
        self.assertEqual(mismatch.kinds, ["mismatch"])
        self.assertNotEqual(mismatch.prompts, ["coastal settlement"])

    def test_short_ddim_sample(self) -> None:
        model = GeoDiffGAN.from_config(self.config).eval()
        lr = torch.rand(1, 3, 16, 16)
        encoder = HashTextEncoder(32, 8)
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = DiagnosticRecorder(
                directory,
                verbose=False,
                panel_size=160,
                save_tensors=True,
            )
            with mock.patch.object(
                model,
                "apply_ablation_inputs",
                wraps=model.apply_ablation_inputs,
            ) as apply_ablation:
                output = model.sample(
                    lr,
                    encoder(["agricultural fields"]),
                    sample_steps=2,
                    null_context=encoder([""]),
                    generator=torch.Generator().manual_seed(4),
                    diagnostics=diagnostics,
                    diffusion_debug_interval=1,
                )
            self.assertEqual(apply_ablation.call_count, 1)
            diagnostics.add_spatial_metrics(
                lr,
                output.base,
                output.residual,
                output.image,
                torch.tensor([[0.4, 0.0, 0.0, 0.0]]),
                scale=4,
                target=output.image,
            )
            report = diagnostics.export()
            self.assertEqual(output.image.shape, (1, 3, 64, 64))
            self.assertTrue(report.exists())
            self.assertTrue((Path(directory) / "overview.png").exists())
            self.assertTrue((Path(directory) / "features.png").exists())
            self.assertTrue((Path(directory) / "diffusion_trajectory.png").exists())
            self.assertTrue((Path(directory) / "projection_trajectory.png").exists())
            self.assertTrue((Path(directory) / "stage_intermediates.png").exists())
            self.assertTrue((Path(directory) / "tensor_histograms.png").exists())
            self.assertTrue((Path(directory) / "frequency_spectra.png").exists())
            self.assertTrue((Path(directory) / "policy_overlays.png").exists())
            self.assertTrue((Path(directory) / "edges_and_wavelets.png").exists())
            self.assertTrue((Path(directory) / "intermediate_tensors.npz").exists())

    def test_diagnostics_detect_nonfinite(self) -> None:
        stats = tensor_statistics(torch.tensor([1.0, float("nan"), float("inf")]))
        self.assertEqual(stats["nan_count"], 1)
        self.assertEqual(stats["inf_count"], 1)
        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticRecorder(directory, verbose=False, fail_on_nonfinite=True)
            with self.assertRaises(FloatingPointError):
                recorder.capture("bad", torch.tensor([float("nan")]))

    def test_degradation_and_projection(self) -> None:
        hr = torch.rand(2, 3, 64, 64)
        lr, parameters = random_degradation(hr, scale=4)
        self.assertEqual(lr.shape, (2, 3, 16, 16))
        clean_lr = sensor_degrade(hr, parameters, scale=4)
        projected = back_project(hr, clean_lr, parameters, scale=4, iterations=1)
        self.assertEqual(projected.shape, hr.shape)
        self.assertTrue(torch.isfinite(projected).all())

    def test_mild_degradation_is_quieter_than_severe(self) -> None:
        hr = torch.full((1, 3, 128, 128), 0.2)
        mild, _, mild_clean = random_degradation(
            hr,
            scale=4,
            generator=torch.Generator().manual_seed(7),
            return_clean=True,
            severity="mild",
        )
        severe, _, severe_clean = random_degradation(
            hr,
            scale=4,
            generator=torch.Generator().manual_seed(7),
            return_clean=True,
            severity="severe",
        )
        mild_ratio = (mild - mild_clean).abs().mean() / mild_clean.mean()
        severe_ratio = (severe - severe_clean).abs().mean() / severe_clean.mean()
        self.assertLess(float(mild_ratio), 0.04)
        self.assertLess(float(mild_ratio), float(severe_ratio))

    def test_evaluation_degradation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patch = root / "patch.npz"
            np.savez_compressed(
                patch,
                hr=np.random.default_rng(3).random((3, 64, 64)).astype(np.float32),
            )
            manifest = root / "manifest.jsonl"
            write_manifest(
                manifest,
                [
                    ManifestRecord(
                        patch=str(patch),
                        tile_id="TEST_TILE",
                        split="test",
                        row=10,
                        col=20,
                        valid_fraction=1.0,
                    )
                ],
            )
            dataset = SentinelPatchDataset(
                manifest,
                split="test",
                augment=False,
                degradation_seed=42,
                degradation_severity="mild",
            )
            first = dataset[0]
            second = dataset[0]
            self.assertTrue(torch.equal(first["lr"], second["lr"]))
            self.assertTrue(torch.equal(first["clean_lr"], second["clean_lr"]))
            self.assertTrue(torch.equal(first["degradation"], second["degradation"]))

    def test_adaptive_edge_f1_handles_low_contrast_edges(self) -> None:
        target = torch.full((1, 3, 32, 32), 0.2)
        target[:, :, :, 16:] = 0.22
        shifted = torch.full_like(target, 0.2)
        shifted[:, :, :, 17:] = 0.22
        self.assertGreater(float(edge_f1(target, target)), 0.99)
        self.assertGreater(float(edge_f1(shifted, target, tolerance=1)), 0.9)

    def test_high_pass_does_not_create_constant_border_edges(self) -> None:
        constant = torch.ones(1, 3, 32, 32)
        filtered = high_pass(constant)
        self.assertLess(float(filtered.abs().max()), 1e-6)

    def test_tile_split_is_stable(self) -> None:
        self.assertEqual(deterministic_split("43PGQ"), deterministic_split("43PGQ"))
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            record = ManifestRecord(
                patch="patch.npz",
                tile_id="43PGQ",
                split=deterministic_split("43PGQ"),
                row=0,
                col=0,
                valid_fraction=1.0,
            )
            write_manifest(manifest, [record])
            loaded = load_manifest(manifest)
            self.assertEqual(loaded[0], record)

    def test_safe_product_prefixes_override_dataset_splits(self) -> None:
        self.assertEqual(
            split_for_product(
                "CHHATARPUR1_scene.SAFE",
                "44QLL",
                validation_prefixes=["CHHATARPUR2"],
                test_prefixes=["chhatarpur1"],
                unmatched_split="train",
            ),
            "test",
        )
        self.assertEqual(
            split_for_product(
                "CHHATARPUR2_scene.SAFE",
                "44QMK",
                validation_prefixes=["chhatarpur2"],
                test_prefixes=["CHHATARPUR1"],
                unmatched_split="train",
            ),
            "val",
        )
        self.assertEqual(
            split_for_product(
                "OTHER_scene.SAFE",
                "44QML",
                validation_prefixes=["CHHATARPUR2"],
                test_prefixes=["CHHATARPUR1"],
                unmatched_split="train",
            ),
            "train",
        )

    def test_safe_product_split_reassignment_prevents_tile_leakage(self) -> None:
        records = [
            ManifestRecord(
                patch="test.npz",
                tile_id="TEST_TILE",
                split="train",
                row=0,
                col=0,
                valid_fraction=1.0,
                source_product="CHHATARPUR1_scene.SAFE",
            ),
            ManifestRecord(
                patch="val.npz",
                tile_id="VAL_TILE",
                split="train",
                row=0,
                col=0,
                valid_fraction=1.0,
                source_product="CHHATARPUR2_scene.SAFE",
            ),
        ]
        reassigned = reassign_product_splits(
            records,
            validation_prefixes=["CHHATARPUR2"],
            test_prefixes=["CHHATARPUR1"],
            unmatched_split="train",
        )
        self.assertEqual([record.split for record in reassigned], ["test", "val"])
        validate_tile_split_isolation(reassigned)

        records[1].tile_id = "TEST_TILE"
        with self.assertRaisesRegex(ValueError, "tile leakage"):
            reassign_product_splits(
                records,
                validation_prefixes=["CHHATARPUR2"],
                test_prefixes=["CHHATARPUR1"],
                unmatched_split="train",
            )

    def test_caption_model_class_resolution(self) -> None:
        explicit = type("ExplicitQwen3VL", (), {})
        automatic = type("AutomaticMultimodal", (), {})
        module = SimpleNamespace(
            Qwen3VLForConditionalGeneration=explicit,
            AutoModelForMultimodalLM=automatic,
        )
        self.assertIs(_resolve_model_class(module), explicit)
        self.assertIs(
            _resolve_model_class(
                SimpleNamespace(AutoModelForMultimodalLM=automatic)
            ),
            automatic,
        )
        with self.assertRaises(RuntimeError):
            _resolve_model_class(SimpleNamespace())

    def test_attention_heads_must_divide_channels(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            WindowTransformerBlock(channels=10, heads=6)
        with self.assertRaisesRegex(ValueError, "divisible"):
            CrossAttention2d(channels=10, context_dim=8, heads=6)

    def test_rgb_window_maps_to_scl_resolution(self) -> None:
        try:
            from affine import Affine
            from rasterio.windows import Window, bounds, from_bounds
        except ImportError:
            self.skipTest("rasterio is not installed")

        rgb_window = Window(col_off=10752, row_off=0, width=228, height=512)
        rgb_transform = Affine(10, 0, 600000, 0, -10, 2400000)
        scl_transform = Affine(20, 0, 600000, 0, -20, 2400000)
        scl_window = from_bounds(
            *bounds(rgb_window, rgb_transform),
            transform=scl_transform,
        )

        self.assertAlmostEqual(scl_window.col_off, 5376)
        self.assertAlmostEqual(scl_window.row_off, 0)
        self.assertAlmostEqual(scl_window.width, 114)
        self.assertAlmostEqual(scl_window.height, 256)


if __name__ == "__main__":
    unittest.main()
