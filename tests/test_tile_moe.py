from __future__ import annotations

import ast
import io
import json
import sys
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from geodiff_gan.data.dataset import SentinelPatchDataset
from geodiff_gan.data.manifest import ManifestRecord, write_manifest
from geodiff_gan.experiments.tile_moe import experiment_config, lock_json, assert_base_lineage, balanced_manifest
from geodiff_gan.models.moe import MixtureDiffusionUNet, router_objectives
from geodiff_gan.models.residual_base import MaskedWindowBlock, ResidualSwinBase
from geodiff_gan.models.system import GeoDiffGAN
from geodiff_gan.training.trainer import Trainer
from geodiff_gan.cli import evaluate, baselines
from geodiff_gan.experiments.tile_viewer import SavedTileResults

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(manifest, root, profile="reliability_moe"):
    config = experiment_config(ROOT, profile, manifest, root, fast=True)
    config["model"].update(base_embed_dim=8, base_heads=2, base_depth=2, base_groups=1,
                          vae_channels=4, lr_channels=4, diffusion_widths=[8, 16],
                          mapper_channels=8, style_dim=8, decoder_channels=[8, 8, 8, 8],
                          context_dim=8, expert_channels=8)
    config["training"].update(num_workers=0, batch_size=1, gradient_accumulation=1,
                             amp=False, epochs=1, progress_mode="quiet", discriminator_channels=4)
    return config


class TileMoETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        lr = np.random.default_rng(4).random((3, 40, 40), dtype=np.float32)
        hr = np.repeat(np.repeat(lr, 3, axis=1), 3, axis=2)
        path = root / "patch.npz"
        np.savez(path, lr=lr, hr=hr, valid_mask_hr=np.ones((1, 120, 120), dtype=np.float32),
                 valid_mask_lr=np.ones((1, 40, 40), dtype=np.float32))
        manifest = root / "manifest.jsonl"
        write_manifest(manifest, [ManifestRecord(str(path), "tile", split, 0, 0, 1.0, scale=3)
                                  for split in ("train", "val", "test")])
        return manifest

    def test_aligned_random_crop_and_eval_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.fixture(Path(directory))
            dataset = SentinelPatchDataset(manifest, "train", scale=3, input_mode="paired", paired_lr_crop_size=32)
            starts = []
            for _ in range(12):
                value = dataset[0]
                self.assertEqual(value["hr"].shape, (3, 96, 96))
                self.assertTrue(torch.equal(value["hr"], value["lr"].repeat_interleave(3, -1).repeat_interleave(3, -2)))
                self.assertEqual(value["valid_mask"].shape, (1, 96, 96))
                starts.append(float(value["lr"][0, 0, 0]))
            self.assertGreater(len(set(starts)), 1)
            for split in ("val", "test"):
                value = SentinelPatchDataset(manifest, split, scale=3, input_mode="paired", paired_lr_crop_size=32, augment=False)[0]
                self.assertEqual(value["lr"].shape[-1], 40)

    def test_window_mask_positions_and_base_anchor(self):
        block = MaskedWindowBlock(8, 2, 8, True)
        mask = block.attention_mask(17, 23, torch.device("cpu"))
        self.assertTrue((mask == -100).any())
        self.assertEqual(int(block.relative_index.min()), 0)
        self.assertEqual(int(block.relative_index.max()), 224)
        self.assertTrue(torch.equal(block.relative_index.diag(), torch.full((64,), 112)))
        x = torch.rand(1, 8, 17, 23, requires_grad=True)
        block(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        model = ResidualSwinBase(embed_dim=8, heads=2, depth=2, groups=1)
        for size in (32, 64, 128):
            lr = torch.rand(1, 3, size, size)
            with torch.no_grad():
                output = model(lr)
            self.assertEqual(output.shape[-1], size * 3)
            self.assertTrue(torch.equal(output, F.interpolate(lr, scale_factor=3, mode="bicubic", align_corners=False).clamp(0, 1)))

    def test_sparse_dense_parity_and_configurable_experts(self):
        for count in (1, 2, 4):
            model = MixtureDiffusionUNet(num_experts=count, top_k=min(count, 2), widths=(8, 16),
                context_dim=8, lr_condition_channels=8, expert_channels=8, upsample_mode="resize_conv")
            args = (torch.rand(2, 4, 12, 12), torch.tensor([20, 400]), torch.zeros(2, 1, 8),
                    torch.zeros(2, 4), torch.zeros(2, dtype=torch.long), torch.rand(2, 8, 16, 16))
            evidence = torch.rand(2, 9, 32, 32)
            model.eval()
            with torch.no_grad():
                sparse = model(*args, routing_context=evidence)
                dense, _, route = model.forward_with_routing(*args, routing_context=evidence)
            self.assertTrue(torch.allclose(sparse, dense, atol=1e-6))
            self.assertTrue(torch.allclose(route["weights"].sum(1), torch.ones(2)))
            self.assertTrue(((route["weights"] > 0).sum(1) <= min(count, 2)).all())
        with self.assertRaises(ValueError):
            MixtureDiffusionUNet(num_experts=2, top_k=3)

    def test_quality_labels_perfect_harmful_zero_and_masked(self):
        logits = torch.zeros(1, 3, requires_grad=True)
        route = {"logits": logits, "quality_logits": logits, "probabilities": logits.softmax(1),
                 "selected": torch.ones_like(logits)}
        candidates = torch.zeros(1, 3, 4, 4, 4, requires_grad=True)
        kwargs = dict(candidate_mse=torch.tensor([[0., .02, .01]]), base_mse=torch.tensor([.01]), temperature=.001)
        losses = router_objectives(route, candidates, candidates[:, 0].detach(), torch.ones(1, 1, 32, 32), **kwargs)
        losses["router_quality"].backward()
        self.assertLess(float(logits.grad[0, 0]), 0)
        self.assertGreater(float(logits.grad[0, 1]), 0)
        self.assertAlmostEqual(float(logits.grad[0, 2]), 0)
        losses = router_objectives(route, candidates, candidates[:, 0].detach(), torch.zeros(1, 1, 32, 32), **kwargs)
        self.assertEqual(float(losses["router_quality"].detach()), 0)

    def test_train_router_and_sampled_refinement_freezes_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            config = tiny_config(manifest, root / "run")
            config["training"].update(stage="diffusion", output_dir=str(root / "diffusion"))
            trainer = Trainer(config)
            self.assertFalse(any(isinstance(module, torch.nn.PixelShuffle) for module in trainer.model.modules()))
            batch = next(iter(trainer._loader("train")))
            _, losses = trainer._forward_stage(batch)
            trainer._weighted_loss(losses).backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.model.diffusion.router.parameters()))
            self.assertTrue(all(p.grad is None for p in trainer.model.base.parameters()))
            config["training"].update(stage="joint", joint_latent_source="sampled", joint_sample_steps=2,
                                     trainable_modules=["mapper", "decoder"])
            trainer = Trainer(config)
            for child in trainer.model.children():
                if not any(p.requires_grad for p in child.parameters()):
                    child.eval()
            _, losses = trainer._forward_stage(batch)
            trainer._weighted_loss(losses).backward()
            self.assertTrue(all(p.grad is None for p in trainer.model.diffusion.parameters()))
            self.assertTrue(any(p.grad is not None for p in trainer.model.decoder.parameters()))
            trainer.train()
            checkpoint = root / "diffusion" / "joint_best.pt"
            payload = torch.load(checkpoint, weights_only=False)
            clone = GeoDiffGAN.from_config(payload["config"])
            clone.load_state_dict(payload["model"], strict=True)
            self.assertIn("stage_elapsed_seconds", payload["extra"])
            assert_base_lineage(checkpoint, checkpoint)

    def test_controlled_generic_reliability_configuration_and_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            configs = [experiment_config(ROOT, name, manifest, root / name) for name in ("generic_moe", "reliability_moe")]
            models = [GeoDiffGAN.from_config(config) for config in configs]
            self.assertEqual(sum(p.numel() for p in models[0].parameters()), sum(p.numel() for p in models[1].parameters()))
            path = root / "lock.json"
            lock_json(path, {"experts": 2})
            lock_json(path, {"experts": 2})
            with self.assertRaises(ValueError):
                lock_json(path, {"experts": 4})

    def test_notebook_ast_and_source_packaging(self):
        path = ROOT / "kaggle/GeoDiff_GAN_Kaggle_Tile_Residual_MoE_3x.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        texts = []
        for cell in notebook["cells"]:
            text = "".join(cell["source"])
            texts.append(text)
            if cell["cell_type"] == "code":
                ast.parse(text)
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        all_text = "\n".join(texts)
        for required in ('REPOSITORY_BRANCH = "3x-continued"', 'geodiff_gan.cli.prepare_landsat_sentinel',
                         'within-tile-spatial', 'RUN_TEST_EVALUATION = False',
                         'run_new("generic_moe")', 'run_new("reliability_moe")'):
            self.assertIn(required, all_text)

    def test_balanced_manifest_keeps_splits_and_interleaves_tiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            rows = [{"patch": f"{tile}/{split}/{i}.npz", "tile_id": tile, "split": split}
                    for split in ("train", "val", "test") for tile in ("A", "B", "C") for i in range(10)]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            destination = balanced_manifest(source, root / "balanced.jsonl")
            balanced = [json.loads(line) for line in destination.read_text().splitlines()]
            self.assertEqual({row["patch"] for row in rows}, {row["patch"] for row in balanced})
            for split in ("train", "val", "test"):
                first = [row for row in balanced if row["split"] == split][:3]
                self.assertEqual({row["tile_id"] for row in first}, {"A", "B", "C"})
            self.assertEqual(destination, balanced_manifest(source, destination))

    def test_complete_miniature_pipeline_and_cached_viewer(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            manifest = self.fixture(root)
            parent = None
            for stage in ("base", "vae", "diffusion", "joint"):
                config = tiny_config(manifest, root / "run")
                config["training"].update(stage=stage, output_dir=str(root / "run" / stage),
                    init_checkpoint=str(parent) if parent else None,
                    validation_limit=1, max_batches_per_epoch=1)
                if stage == "diffusion":
                    config["training"].update(checkpoint_metric="val_loss_diffusion", checkpoint_mode="min")
                if stage == "joint":
                    config["training"].update(joint_latent_source="sampled", joint_sample_steps=2,
                                             trainable_modules=["mapper", "decoder"])
                Trainer(config).train()
                checkpoint = root / "run" / stage / f"{stage}_best.pt"
                self.assertTrue(checkpoint.is_file())
                if parent:
                    assert_base_lineage(parent, checkpoint)
                parent = checkpoint
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            output = root / "run/evaluation/test/model"
            arguments = ["evaluate", "--config", str(config_path), "--checkpoint", str(parent),
                "--output", str(output), "--split", "test", "--samples", "2", "--steps", "2",
                "--back-projection-steps", "0", "--device", "cpu", "--no-text", "--progress", "quiet"]
            with mock.patch.object(sys, "argv", arguments):
                evaluate.main()
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["count"], 1)
            self.assertTrue(np.isfinite(metrics["psnr"]))
            with mock.patch.object(sys, "argv", ["baselines", "--config", str(config_path),
                "--base-checkpoint", str(parent), "--output", str(root / "baseline.json"),
                "--device", "cpu", "--split", "test", "--progress", "quiet"]):
                baselines.main()
            baseline_metrics = json.loads((root / "baseline.json").read_text())
            self.assertAlmostEqual(baseline_metrics["base"]["psnr"], metrics["base_psnr"], places=4)
            (root / "suite_state.json").write_text(json.dumps({"manifest": str(manifest),
                "results": {"model": {"root": str(root / "run")}}}))
            viewer = SavedTileResults(root)
            panels, info = viewer.panels(0, show_base=True)
            self.assertEqual(len(panels), 4)
            self.assertEqual(len(info["models"]["model"]["expert_weights"]), 2)
            self.assertTrue(0 <= info["models"]["model"]["router_acceptance"] <= 1)
            self.assertEqual(panels[0][0].shape[-1], 40)
            self.assertEqual(panels[-1][0].shape[-1], 120)


if __name__ == "__main__":
    unittest.main()
