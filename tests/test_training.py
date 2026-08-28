from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from geodiff_gan.config import load_config
from geodiff_gan.data.manifest import ManifestRecord, write_manifest
from geodiff_gan.training import Trainer
from geodiff_gan.training.checkpoint import (
    best_stage_checkpoint,
    latest_stage_checkpoint,
    prune_stage_epoch_checkpoints,
)


ROOT = Path(__file__).resolve().parents[1]


class TrainingSmokeTest(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        record_count: int = 1,
        include_validation: bool = False,
    ) -> tuple[Path, dict]:
        patch = root / "patch.npz"
        np.savez_compressed(
            patch,
            hr=np.random.default_rng(2).random((3, 64, 64)).astype(np.float32),
        )
        manifest = root / "manifest.jsonl"
        records = [
            ManifestRecord(
                patch=str(patch),
                tile_id="TEST_TILE",
                split="train",
                row=index,
                col=0,
                valid_fraction=1.0,
                caption="mixed agricultural fields",
            )
            for index in range(record_count)
        ]
        if include_validation:
            records.append(
                ManifestRecord(
                    patch=str(patch),
                    tile_id="VALIDATION_TILE",
                    split="val",
                    row=0,
                    col=0,
                    valid_fraction=1.0,
                    caption="mixed agricultural fields",
                )
            )
        write_manifest(manifest, records)
        config = load_config(ROOT / "configs/smoke.yaml", ROOT / "configs/default.yaml")
        config["data"]["manifest"] = str(manifest)
        config["data"]["captions"] = None
        config["training"].update(
            {
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation": 1,
                "num_workers": 0,
                "init_checkpoint": None,
                "resume": None,
                "auto_resume": False,
                "progress_mode": "quiet",
                "progress_updates_per_epoch": 0,
                "validation_limit": 1,
            }
        )
        return manifest, config

    def test_base_stage_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = self._fixture(root, include_validation=True)
            config["training"].update(
                {
                    "stage": "base",
                    "output_dir": str(root / "run"),
                }
            )
            Trainer(config).train()
            self.assertTrue((root / "run" / "base_epoch_0000.pt").exists())
            self.assertEqual(
                best_stage_checkpoint(root / "run", "base"),
                root / "run" / "base_best.pt",
            )
            self.assertTrue((root / "run" / "training_history.jsonl").exists())
            self.assertTrue((root / "run" / "training_curves.png").exists())
            self.assertTrue((root / "run" / "latest_metrics.json").exists())
            latest = json.loads(
                (root / "run" / "latest_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            for metric in (
                "val_l1",
                "val_psnr",
                "val_ssim",
                "val_edge_f1",
                "val_redegradation_l1",
            ):
                self.assertIn(metric, latest["metrics"])

    def test_nonbase_stage_can_disable_text_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = self._fixture(root)
            config["model"]["use_text_conditioning"] = False
            config["training"].update(
                {
                    "stage": "diffusion",
                    "output_dir": str(root / "run"),
                }
            )
            with mock.patch(
                "geodiff_gan.training.trainer.build_text_encoder"
            ) as build_text_encoder:
                trainer = Trainer(config)
            build_text_encoder.assert_not_called()
            context, null_context, prompts, kinds = trainer._contexts(
                ["caption-free sample"],
                training=False,
            )
            self.assertEqual(tuple(context.shape), (1, 1, trainer.model.context_dim))
            self.assertTrue(torch.equal(context, null_context))
            self.assertEqual(prompts, ["caption-free sample"])
            self.assertEqual(kinds, ["original"])

    def test_latest_stage_checkpoint_uses_numeric_epoch_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "base_epoch_0002.pt",
                "base_epoch_0010.pt",
                "vae_epoch_0099.pt",
            ):
                (root / name).touch()
            latest = latest_stage_checkpoint(root, "base")
            self.assertEqual(latest, root / "base_epoch_0010.pt")

    def test_checkpoint_pruning_keeps_only_latest_epoch_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = [
                root / "base_epoch_0000.pt",
                root / "base_epoch_0001.pt",
                root / "base_epoch_0002.pt",
            ]
            for checkpoint in checkpoints:
                checkpoint.touch()
            (root / "base_best.pt").touch()

            prune_stage_epoch_checkpoints(
                root,
                "base",
                keep=checkpoints[-1],
            )

            self.assertEqual(
                sorted(path.name for path in root.glob("base*.pt")),
                ["base_best.pt", "base_epoch_0002.pt"],
            )

    def test_auto_resume_continues_from_next_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = self._fixture(root)
            output = root / "resume"
            config["training"].update(
                {
                    "stage": "base",
                    "output_dir": str(output),
                }
            )
            Trainer(config).train()

            resumed = config.copy()
            resumed["data"] = config["data"].copy()
            resumed["training"] = config["training"].copy()
            resumed["training"].update(
                {
                    "epochs": 2,
                    "resume": None,
                    "auto_resume": True,
                }
            )
            trainer = Trainer(resumed)
            self.assertEqual(trainer.start_epoch, 1)
            trainer.train()

            self.assertTrue((output / "base_epoch_0001.pt").exists())
            self.assertFalse((output / "base_epoch_0000.pt").exists())
            self.assertTrue((output / "base_best.pt").exists())
            history = [
                json.loads(line)
                for line in (output / "training_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([entry["epoch"] for entry in history], [0, 1])

    def test_early_stopping_uses_validation_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = self._fixture(root, include_validation=True)
            output = root / "early_stop"
            config["training"].update(
                {
                    "stage": "base",
                    "epochs": 8,
                    "output_dir": str(output),
                    "early_stopping_patience": 2,
                    "early_stopping_min_epochs": 2,
                    "early_stopping_min_delta": 0.0,
                }
            )
            trainer = Trainer(config)
            validation_results = [
                {"val_l1": 0.1},
                {"val_l1": 0.11},
                {"val_l1": 0.12},
            ]
            with mock.patch.object(
                trainer,
                "_validate",
                side_effect=validation_results,
            ):
                trainer.train()

            history = [
                json.loads(line)
                for line in (output / "training_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(history), 3)
            self.assertTrue((output / "base_best.pt").exists())
            self.assertTrue((output / "base_epoch_0002.pt").exists())
            self.assertFalse((output / "base_epoch_0001.pt").exists())

    def test_early_stopping_ignores_epochs_without_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = self._fixture(root, include_validation=True)
            output = root / "sparse_validation"
            config["training"].update(
                {
                    "stage": "base",
                    "epochs": 3,
                    "output_dir": str(output),
                    "validate_every": 2,
                    "early_stopping_patience": 1,
                    "early_stopping_min_epochs": 1,
                }
            )
            trainer = Trainer(config)
            with mock.patch.object(
                trainer,
                "_validate",
                return_value={"val_l1": 0.1},
            ) as validate:
                trainer.train()

            self.assertEqual(validate.call_count, 1)
            history = [
                json.loads(line)
                for line in (output / "training_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(history), 3)
            self.assertTrue((output / "base_best.pt").exists())
            self.assertTrue((output / "base_epoch_0002.pt").exists())

    def test_remaining_training_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, base_config = self._fixture(root)
            for stage in ("vae", "diffusion", "joint", "edit"):
                with self.subTest(stage=stage):
                    config = load_config(
                        ROOT / "configs/smoke.yaml", ROOT / "configs/default.yaml"
                    )
                    config.update(base_config)
                    config["data"] = base_config["data"].copy()
                    config["training"] = base_config["training"].copy()
                    output = root / stage
                    config["training"].update({"stage": stage, "output_dir": str(output)})
                    Trainer(config).train()
                    self.assertTrue((output / f"{stage}_epoch_0000.pt").exists())

    def test_partial_accumulation_group_still_updates_generator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = self._fixture(root)
            config["training"].update(
                {
                    "stage": "base",
                    "output_dir": str(root / "partial"),
                    "gradient_accumulation": 2,
                }
            )
            trainer = Trainer(config)
            before = next(trainer.model.base.parameters()).detach().clone()
            trainer.train()
            after = next(trainer.model.base.parameters()).detach()
            self.assertFalse(torch.equal(before, after))

    def test_joint_accumulates_discriminator_and_uses_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = self._fixture(root, record_count=2)
            config["training"].update(
                {
                    "stage": "joint",
                    "output_dir": str(root / "joint_accumulation"),
                    "gradient_accumulation": 2,
                    "train_back_projection_steps": 1,
                }
            )
            trainer = Trainer(config)
            batch = next(iter(trainer._loader("train")))
            with mock.patch.object(
                trainer.model,
                "decode_latent",
                wraps=trainer.model.decode_latent,
            ) as decode:
                prediction, _ = trainer._forward_stage(batch)
            self.assertEqual(decode.call_args.kwargs["back_projection_steps"], 1)

            adversarial = trainer._generator_adversarial_loss(
                prediction,
                batch["lr"].to(trainer.device),
            )
            adversarial.backward()
            discriminator_parameters = list(
                trainer.patch_discriminator.parameters()
            ) + list(trainer.wavelet_discriminator.parameters())
            self.assertTrue(
                all(parameter.grad is None for parameter in discriminator_parameters)
            )
            trainer.optimizer.zero_grad(set_to_none=True)

            with mock.patch.object(
                trainer,
                "_discriminator_loss",
                wraps=trainer._discriminator_loss,
            ) as discriminator_loss:
                trainer.train()
            self.assertEqual(discriminator_loss.call_count, 2)

    def test_trainable_modules_override_can_unfreeze_base_for_joint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = self._fixture(root)
            config["training"].update(
                {
                    "stage": "joint",
                    "output_dir": str(root / "joint_default"),
                }
            )
            default_trainer = Trainer(config)
            self.assertFalse(
                any(parameter.requires_grad for parameter in default_trainer.model.base.parameters())
            )
            self.assertTrue(
                any(parameter.requires_grad for parameter in default_trainer.model.decoder.parameters())
            )

            config["training"].update(
                {
                    "output_dir": str(root / "joint_unfrozen"),
                    "trainable_modules": [
                        "base",
                        "vae",
                        "lr_encoder",
                        "diffusion",
                        "mapper",
                        "decoder",
                    ],
                }
            )
            unfrozen_trainer = Trainer(config)
            self.assertTrue(
                any(parameter.requires_grad for parameter in unfrozen_trainer.model.base.parameters())
            )
            self.assertTrue(
                any(parameter.requires_grad for parameter in unfrozen_trainer.model.vae.parameters())
            )

    def test_joint_supports_module_learning_rate_multipliers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = self._fixture(root)
            config["training"].update(
                {
                    "stage": "joint",
                    "output_dir": str(root / "joint_lrs"),
                    "learning_rate": 1e-4,
                    "module_learning_rate_multipliers": {
                        "diffusion": 0.25,
                        "decoder": 1.0,
                    },
                }
            )
            trainer = Trainer(config)
            learning_rates = {
                group.get("name"): group["lr"]
                for group in trainer.optimizer.param_groups
            }
            self.assertAlmostEqual(learning_rates["diffusion"], 2.5e-5)
            self.assertAlmostEqual(learning_rates["decoder"], 1e-4)

    def test_joint_trains_trust_controller_and_biases_reconstruction_timesteps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = self._fixture(root)
            config["model"].update(
                {
                    "use_base_referenced_trust": True,
                    "trust_channels": 8,
                    "trust_initial_scale": 0.25,
                }
            )
            config["training"].update(
                {
                    "stage": "joint",
                    "output_dir": str(root / "joint_trust"),
                    "joint_max_timestep_fraction": 0.25,
                }
            )
            trainer = Trainer(config)
            self.assertTrue(
                any(
                    parameter.requires_grad
                    for parameter in trainer.model.trust_controller.parameters()
                )
            )
            batch = next(iter(trainer._loader("train")))
            with mock.patch.object(
                trainer.model,
                "prepare_diffusion_batch",
                wraps=trainer.model.prepare_diffusion_batch,
            ) as prepare:
                trainer._forward_stage(batch)
            timesteps = prepare.call_args.kwargs["timesteps"]
            self.assertIsNotNone(timesteps)
            self.assertLess(int(timesteps.max()), 250)

    def test_diffusion_checkpoint_fallback_minimizes_validation_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = self._fixture(root, include_validation=True)
            output = root / "diffusion_fallback"
            config["training"].update(
                {
                    "stage": "diffusion",
                    "epochs": 2,
                    "output_dir": str(output),
                    "checkpoint_metric": "val_psnr",
                    "checkpoint_mode": "max",
                    "early_stopping_patience": 0,
                }
            )
            trainer = Trainer(config)
            with mock.patch.object(
                trainer,
                "_validate",
                side_effect=[
                    {"val_loss_total": 2.0},
                    {"val_loss_total": 1.0},
                ],
            ):
                trainer.train()
            payload = torch.load(
                output / "diffusion_best.pt",
                map_location="cpu",
                weights_only=False,
            )
            selection = payload["extra"]["checkpoint_selection"]
            self.assertEqual(payload["epoch"], 1)
            self.assertEqual(selection["metric"], "val_loss_total")
            self.assertEqual(selection["mode"], "min")
            self.assertEqual(selection["value"], 1.0)

    def test_zero_adversarial_weight_skips_discriminators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = self._fixture(root)
            config["training"].update(
                {
                    "stage": "joint",
                    "output_dir": str(root / "no_gan"),
                }
            )
            config["training"]["loss_weights"]["adversarial"] = 0.0
            trainer = Trainer(config)
            with mock.patch.object(
                trainer,
                "_discriminator_loss",
                side_effect=AssertionError("discriminator should be skipped"),
            ):
                trainer.train()

    def test_counterfactual_edit_does_not_use_paired_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = self._fixture(root)
            config["training"].update(
                {
                    "stage": "edit",
                    "output_dir": str(root / "counterfactual"),
                }
            )
            config["prompts"] = {
                "null_probability": 0.0,
                "paraphrase_probability": 0.0,
                "mismatch_probability": 1.0,
            }
            trainer = Trainer(config)
            batch = next(iter(trainer._loader("train")))
            _, losses = trainer._forward_stage(batch)
            self.assertEqual(float(losses["charbonnier"]), 0.0)
            self.assertEqual(float(losses["ssim"]), 0.0)
            self.assertIn("edit_permission", losses)
            self.assertIn("edit_localization", losses)


if __name__ == "__main__":
    unittest.main()
