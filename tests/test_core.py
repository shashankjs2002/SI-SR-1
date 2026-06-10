from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from geodiff_gan.config import load_config
from geodiff_gan.data.manifest import ManifestRecord, deterministic_split, load_manifest, write_manifest
from geodiff_gan.diagnostics import DiagnosticRecorder, tensor_statistics
from geodiff_gan.models.blocks import high_pass
from geodiff_gan.models.degradation import back_project, random_degradation, sensor_degrade
from geodiff_gan.models.system import GeoDiffGAN
from geodiff_gan.text import HashTextEncoder


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
        self.assertFalse(output.metadata[0]["synthetic_edit"])
        edit = model.decode_latent(
            latent, lr, context, degradation, mode="edit", base=base, back_projection_steps=0
        )
        self.assertTrue(edit.metadata[0]["synthetic_edit"])

    def test_short_ddim_sample(self) -> None:
        model = GeoDiffGAN.from_config(self.config).eval()
        lr = torch.rand(1, 3, 16, 16)
        encoder = HashTextEncoder(32, 8)
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = DiagnosticRecorder(directory, verbose=False)
            output = model.sample(
                lr,
                encoder(["agricultural fields"]),
                sample_steps=2,
                null_context=encoder([""]),
                generator=torch.Generator().manual_seed(4),
                diagnostics=diagnostics,
                diffusion_debug_interval=1,
            )
            diagnostics.add_spatial_metrics(
                lr,
                output.base,
                output.residual,
                output.image,
                torch.tensor([[0.4, 0.0, 0.0, 0.0]]),
                scale=4,
            )
            report = diagnostics.export()
            self.assertEqual(output.image.shape, (1, 3, 64, 64))
            self.assertTrue(report.exists())
            self.assertTrue((Path(directory) / "overview.png").exists())
            self.assertTrue((Path(directory) / "features.png").exists())
            self.assertTrue((Path(directory) / "diffusion_trajectory.png").exists())

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


if __name__ == "__main__":
    unittest.main()
