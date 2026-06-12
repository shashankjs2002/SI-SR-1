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
from geodiff_gan.training.checkpoint import latest_stage_checkpoint


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
            history = [
                json.loads(line)
                for line in (output / "training_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([entry["epoch"] for entry in history], [0, 1])

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
