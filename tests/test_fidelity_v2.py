from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from geodiff_gan.config import load_config
from geodiff_gan.data import SentinelPatchDataset
from geodiff_gan.data.manifest import ManifestRecord, write_manifest
from geodiff_gan.fidelity import (
    fidelity_acceptance_report,
    paired_bootstrap_interval,
    radiometric_ceiling_report,
    residual_oracle_variants,
)
from geodiff_gan.losses import oracle_residual_trust
from geodiff_gan.models.base import SwinIRV2Block
from geodiff_gan.models.generator import BaseReferencedTrustController
from geodiff_gan.models.system import GeoDiffGAN
from geodiff_gan.training import Trainer
from geodiff_gan.training.checkpoint import load_checkpoint, save_checkpoint
from geodiff_gan.training.ema import ModelEMA


ROOT = Path(__file__).resolve().parents[1]


class FidelityV2Tests(unittest.TestCase):
    def _model_config(self, architecture: str) -> dict:
        config = load_config(
            ROOT / "configs/smoke.yaml",
            ROOT / "configs/default.yaml",
        )
        config["model"].update(
            {
                "scale": 3,
                "base_architecture": architecture,
                "base_embed_dim": 24,
                "base_heads": 6,
                "base_upsample_mode": "resize_conv",
                "base_rdn_blocks": 2,
                "base_rdn_layers": 2,
                "base_rdn_growth": 8,
                "base_swin_groups": 2,
                "base_swin_blocks_per_group": 2,
                "vae_upsample_mode": "resize_conv",
                "diffusion_upsample_mode": "resize_conv",
                "decoder_upsample_mode": "resize_conv",
                "use_base_referenced_trust": True,
                "trust_mode": "per_band",
            }
        )
        return config

    def test_fidelity_bases_start_at_bicubic_and_have_no_pixelshuffle(self) -> None:
        lr = torch.rand(1, 3, 16, 16)
        expected = torch.nn.functional.interpolate(
            lr,
            scale_factor=3,
            mode="bicubic",
            align_corners=False,
        ).clamp(0, 1)
        for architecture in ("fidelity_rdn", "fidelity_swinir_v2"):
            with self.subTest(architecture=architecture):
                model = GeoDiffGAN.from_config(
                    self._model_config(architecture)
                ).eval()
                actual = model.predict_base(lr)
                self.assertEqual(tuple(actual.shape), (1, 3, 48, 48))
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
                self.assertFalse(
                    any(isinstance(module, nn.PixelShuffle) for module in model.modules())
                )

    def test_shifted_window_mask_and_relative_position_index(self) -> None:
        block = SwinIRV2Block(
            channels=24,
            window_size=8,
            heads=6,
            shift=True,
        )
        mask = block.attention_mask(
            16,
            16,
            torch.device("cpu"),
            torch.float32,
        )
        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(tuple(mask.shape), (4, 64, 64))
        index = block.attention.relative_position_index
        self.assertEqual(tuple(index.shape), (64, 64))
        self.assertGreaterEqual(int(index.min()), 0)
        self.assertLess(int(index.max()), (2 * 8 - 1) ** 2)
        output = block(torch.rand(2, 24, 19, 21))
        self.assertEqual(tuple(output.shape), (2, 24, 19, 21))

    def test_per_band_trust_is_bounded_and_identity_initialized(self) -> None:
        controller = BaseReferencedTrustController(
            content_channels=8,
            lr_channels=6,
            hidden_channels=4,
            initial_scale=0.25,
            maximum_scale=1.0,
            mode="per_band",
            output_channels=3,
        ).eval()
        trust = controller(
            torch.rand(1, 8, 4, 4),
            torch.rand(1, 6, 8, 8),
            torch.rand(1, 1, 4, 4),
            torch.rand(1, 3, 32, 32),
            torch.rand(1, 3, 32, 32) - 0.5,
            consistency_error=torch.rand(1, 3, 8, 8),
            sample_variance=torch.rand(1, 1, 32, 32),
        )
        self.assertEqual(tuple(trust.shape), (1, 3, 32, 32))
        self.assertTrue(torch.allclose(trust, torch.full_like(trust, 0.25)))
        self.assertGreaterEqual(float(trust.detach().min()), 0.0)
        self.assertLessEqual(float(trust.detach().max()), 1.0)

    def test_per_band_oracle_rejects_harmful_and_zero_channels(self) -> None:
        base = torch.zeros(1, 3, 16, 16)
        residual = torch.ones_like(base)
        target = torch.empty_like(base)
        target[:, 0].fill_(0.2)
        target[:, 1].fill_(0.6)
        target[:, 2].zero_()
        trust = oracle_residual_trust(
            residual,
            base,
            target,
            smoothing_window=5,
            per_band=True,
            ridge=1e-8,
        )
        expected = torch.tensor([0.2, 0.6, 0.0])[None, :, None, None]
        self.assertTrue(torch.allclose(trust, expected.expand_as(trust), atol=1e-5))
        variants = residual_oracle_variants(
            base,
            residual,
            target,
            smoothing_window=5,
        )
        self.assertTrue(
            torch.allclose(variants["local_per_band"]["image"], target, atol=1e-5)
        )
        zero_trust = oracle_residual_trust(
            torch.zeros_like(residual),
            base,
            target,
            per_band=True,
        )
        self.assertTrue(torch.isfinite(zero_trust).all())
        self.assertTrue(torch.equal(zero_trust, torch.zeros_like(zero_trust)))

        masked_target = target.clone()
        masked_target[:, :, :4, :4] = 1.0
        mask = torch.ones(1, 1, 16, 16)
        mask[:, :, :4, :4] = 0
        masked_trust = oracle_residual_trust(
            residual,
            base,
            masked_target,
            smoothing_window=1,
            per_band=True,
            ridge=1e-8,
            mask=mask,
        )
        self.assertTrue(
            torch.equal(masked_trust[:, :, :4, :4], torch.zeros(1, 3, 4, 4))
        )
        self.assertTrue(
            torch.allclose(
                masked_trust[:, :, 4:, 4:],
                expected.expand_as(masked_trust)[:, :, 4:, 4:],
                atol=1e-5,
            )
        )

    def test_paired_crop_preserves_exact_three_x_alignment_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coordinate = np.arange(80 * 80, dtype=np.float32).reshape(1, 80, 80)
            lr = np.concatenate(
                [coordinate / coordinate.max()] * 3,
                axis=0,
            )
            hr = np.repeat(np.repeat(lr, 3, axis=1), 3, axis=2)
            lr_mask = np.ones((1, 80, 80), dtype=np.float32)
            lr_mask[:, :4, :] = 0
            hr_mask = np.repeat(np.repeat(lr_mask, 3, axis=1), 3, axis=2)
            patch = root / "paired.npz"
            np.savez_compressed(
                patch,
                lr=lr,
                clean_lr=lr,
                hr=hr,
                valid_mask_lr=lr_mask,
                valid_mask_hr=hr_mask,
            )
            manifest = root / "manifest.jsonl"
            write_manifest(
                manifest,
                [
                    ManifestRecord(
                        patch=str(patch),
                        tile_id="PAIR",
                        split="train",
                        row=0,
                        col=0,
                        valid_fraction=1.0,
                    )
                ],
            )
            for crop in (32, 64):
                dataset = SentinelPatchDataset(
                    manifest,
                    split="train",
                    scale=3,
                    input_mode="paired",
                    paired_lr_crop_size=crop,
                    augment=False,
                    random_degradation=False,
                )
                sample = dataset[0]
                self.assertEqual(tuple(sample["lr"].shape), (3, crop, crop))
                self.assertEqual(tuple(sample["hr"].shape), (3, crop * 3, crop * 3))
                self.assertTrue(
                    torch.allclose(sample["hr"][:, ::3, ::3], sample["lr"])
                )
                self.assertTrue(
                    torch.equal(
                        sample["valid_mask"][:, ::3, ::3],
                        sample["valid_mask_lr"],
                    )
                )

    def test_oracle_reports_and_acceptance_gate(self) -> None:
        target = torch.rand(4, 3, 32, 32) * 0.7
        base = (target * 0.8 + 0.05).clamp(0, 1)
        report = radiometric_ceiling_report(
            base,
            target,
            local_window=5,
        )
        self.assertGreater(report["mean_affine_gain_db"], 0)
        lower, upper = paired_bootstrap_interval([0.1, 0.2, 0.3], samples=200)
        self.assertGreaterEqual(lower, 0)
        self.assertGreaterEqual(upper, lower)
        acceptance = fidelity_acceptance_report(
            previous_base_psnr=34.0,
            new_base_psnr=34.8,
            final_psnr=34.9,
            base_ssim=0.90,
            final_ssim=0.901,
            per_image_psnr_delta=[0.1, 0.2, 0.05, 0.3],
            bootstrap_samples=200,
        )
        self.assertTrue(acceptance["passed"])

    def test_checkpoint_can_load_raw_or_ema_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(1.0)
            ema = ModelEMA(model, decay=0.5)
            with torch.no_grad():
                model.weight.fill_(3.0)
            ema.update(model)
            path = Path(directory) / "ema.pt"
            save_checkpoint(
                path,
                model,
                optimizer=None,
                epoch=0,
                stage="base",
                config={},
                extra={"ema": ema.state_dict()},
            )
            raw = nn.Linear(2, 1, bias=False)
            averaged = nn.Linear(2, 1, bias=False)
            load_checkpoint(path, raw, prefer_ema=False)
            load_checkpoint(path, averaged, prefer_ema=True)
            self.assertTrue(torch.allclose(raw.weight, torch.full_like(raw.weight, 3.0)))
            self.assertTrue(
                torch.allclose(averaged.weight, torch.full_like(averaged.weight, 2.0))
            )

    def test_step_budget_scheduler_and_ema_are_checkpointed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patch = root / "patch.npz"
            np.savez_compressed(
                patch,
                hr=np.random.default_rng(8).random((3, 64, 64)).astype(np.float32),
            )
            manifest = root / "manifest.jsonl"
            write_manifest(
                manifest,
                [
                    ManifestRecord(
                        patch=str(patch),
                        tile_id="STEP_BUDGET",
                        split="train",
                        row=index,
                        col=0,
                        valid_fraction=1.0,
                    )
                    for index in range(2)
                ],
            )
            config = load_config(
                ROOT / "configs/smoke.yaml",
                ROOT / "configs/default.yaml",
            )
            output = root / "run"
            config["data"].update({"manifest": str(manifest), "captions": None})
            config["model"]["use_text_conditioning"] = False
            config["training"].update(
                {
                    "stage": "base",
                    "output_dir": str(output),
                    "epochs": 1,
                    "max_optimizer_steps": 3,
                    "batch_size": 1,
                    "gradient_accumulation": 1,
                    "num_workers": 0,
                    "init_checkpoint": None,
                    "resume": None,
                    "auto_resume": False,
                    "progress_mode": "quiet",
                    "progress_updates_per_epoch": 0,
                    "validate_every": 1,
                    "validation_limit": 1,
                    "lr_scheduler_type": "cosine_warmup",
                    "warmup_steps": 1,
                    "ema_decay": 0.9,
                }
            )
            Trainer(config).train()
            payload = torch.load(
                output / "base_epoch_0001.pt",
                map_location="cpu",
                weights_only=False,
            )
            extra = payload["extra"]
            self.assertEqual(extra["optimizer_step"], 3)
            self.assertIn("lr_scheduler", extra)
            self.assertIn("ema", extra)
            history = (output / "training_history.jsonl").read_text().splitlines()
            self.assertEqual(len(history), 2)

    def test_wall_time_budget_stops_after_a_complete_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patch = root / "patch.npz"
            np.savez_compressed(
                patch,
                hr=np.random.default_rng(9).random((3, 64, 64)).astype(np.float32),
            )
            manifest = root / "manifest.jsonl"
            write_manifest(
                manifest,
                [
                    ManifestRecord(
                        patch=str(patch),
                        tile_id="WALL_BUDGET",
                        split="train",
                        row=index,
                        col=0,
                        valid_fraction=1.0,
                    )
                    for index in range(4)
                ],
            )
            config = load_config(
                ROOT / "configs/smoke.yaml",
                ROOT / "configs/default.yaml",
            )
            output = root / "run"
            config["data"].update({"manifest": str(manifest), "captions": None})
            config["model"]["use_text_conditioning"] = False
            config["training"].update(
                {
                    "stage": "base",
                    "output_dir": str(output),
                    "epochs": 1,
                    "max_optimizer_steps": 100,
                    "max_wall_time_minutes": 1e-9,
                    "batch_size": 1,
                    "gradient_accumulation": 1,
                    "num_workers": 0,
                    "init_checkpoint": None,
                    "resume": None,
                    "auto_resume": False,
                    "progress_mode": "quiet",
                    "validate_every": 100,
                    "lr_scheduler_type": "cosine_warmup",
                    "warmup_steps": 1,
                }
            )
            Trainer(config).train()
            checkpoint = output / "base_epoch_0000.pt"
            self.assertTrue(checkpoint.exists())
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["extra"]["optimizer_step"], 1)
            self.assertTrue(payload["extra"]["early_stopping"]["stopped"])

    def test_sampled_joint_freezes_base_and_diffusion_but_updates_heads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patch = root / "patch.npz"
            np.savez_compressed(
                patch,
                hr=np.random.default_rng(4).random((3, 64, 64)).astype(np.float32),
            )
            manifest = root / "manifest.jsonl"
            write_manifest(
                manifest,
                [
                    ManifestRecord(
                        patch=str(patch),
                        tile_id="SAMPLED",
                        split="train",
                        row=0,
                        col=0,
                        valid_fraction=1.0,
                        caption="",
                    )
                ],
            )
            config = load_config(
                ROOT / "configs/smoke.yaml",
                ROOT / "configs/default.yaml",
            )
            config["data"].update({"manifest": str(manifest), "captions": None})
            config["model"].update(
                {
                    "use_text_conditioning": False,
                    "use_base_referenced_trust": True,
                    "trust_mode": "per_band",
                    "trust_channels": 4,
                    "base_upsample_mode": "resize_conv",
                    "vae_upsample_mode": "resize_conv",
                    "diffusion_upsample_mode": "resize_conv",
                    "decoder_upsample_mode": "resize_conv",
                }
            )
            config["training"].update(
                {
                    "stage": "joint",
                    "output_dir": str(root / "run"),
                    "init_checkpoint": None,
                    "resume": None,
                    "num_workers": 0,
                    "batch_size": 1,
                    "joint_latent_source": "sampled",
                    "joint_sample_steps": 1,
                    "trust_samples": 2,
                    "train_back_projection_steps": 0,
                    "trainable_modules": [
                        "mapper",
                        "decoder",
                        "trust_controller",
                    ],
                }
            )
            trainer = Trainer(config)
            batch = next(iter(trainer._loader("train")))
            _, losses = trainer._forward_stage(batch)
            trainer._weighted_loss(losses).backward()

            def has_gradient(module: nn.Module) -> bool:
                return any(
                    parameter.grad is not None
                    and bool(torch.isfinite(parameter.grad).all())
                    for parameter in module.parameters()
                )

            self.assertFalse(has_gradient(trainer.model.base))
            self.assertFalse(has_gradient(trainer.model.diffusion))
            self.assertTrue(has_gradient(trainer.model.mapper))
            self.assertTrue(has_gradient(trainer.model.decoder))
            self.assertTrue(has_gradient(trainer.model.trust_controller))


if __name__ == "__main__":
    unittest.main()
