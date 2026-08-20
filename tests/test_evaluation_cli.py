from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from geodiff_gan.cli import evaluate
from geodiff_gan.config import load_config
from geodiff_gan.data.manifest import ManifestRecord, write_manifest
from geodiff_gan.models.system import GeoDiffGAN
from geodiff_gan.training.checkpoint import save_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class EvaluationCliTest(unittest.TestCase):
    def test_no_text_cpu_evaluation_reports_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patch = root / "patch.npz"
            selected_patch = root / "selected_patch.npz"
            np.savez_compressed(
                patch,
                hr=np.random.default_rng(5).random((3, 64, 64)).astype(np.float32),
            )
            np.savez_compressed(
                selected_patch,
                hr=np.random.default_rng(6).random((3, 64, 64)).astype(np.float32),
            )
            manifest = root / "manifest.jsonl"
            write_manifest(
                manifest,
                [
                    ManifestRecord(
                        patch=str(patch),
                        tile_id="VAL_TILE",
                        split="val",
                        row=0,
                        col=0,
                        valid_fraction=1.0,
                        caption="",
                    ),
                    ManifestRecord(
                        patch=str(selected_patch),
                        tile_id="VAL_TILE",
                        split="val",
                        row=64,
                        col=0,
                        valid_fraction=1.0,
                        caption="",
                    ),
                ],
            )
            config = load_config(
                ROOT / "configs/smoke.yaml",
                ROOT / "configs/default.yaml",
            )
            config["data"]["manifest"] = str(manifest)
            config["data"]["captions"] = None
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            checkpoint = root / "joint_epoch_0000.pt"
            save_checkpoint(
                checkpoint,
                GeoDiffGAN.from_config(config),
                optimizer=None,
                epoch=0,
                stage="joint",
                config=config,
            )
            output = root / "evaluation"
            arguments = [
                "geodiff-evaluate",
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--split",
                "val",
                "--samples",
                "1",
                "--steps",
                "1",
                "--index",
                "1",
                "--device",
                "cpu",
                "--progress",
                "compact",
                "--no-text",
                "--residual-scale",
                "0",
            ]
            stream = io.StringIO()
            with (
                mock.patch.object(sys, "argv", arguments),
                mock.patch.object(
                    evaluate,
                    "build_text_encoder",
                    side_effect=AssertionError("text encoder should not load"),
                ),
                redirect_stdout(stream),
            ):
                evaluate.main()

            metrics = json.loads(
                (output / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["count"], 1)
            self.assertEqual(metrics["samples_per_patch"], 1)
            self.assertEqual(metrics["diffusion_steps"], 1)
            self.assertEqual(metrics["device"], "cpu")
            self.assertFalse(metrics["amp"])
            self.assertFalse(metrics["text_conditioning"])
            self.assertEqual(metrics["residual_scale"], 0.0)
            self.assertEqual(metrics["dataset_index"], 1)
            self.assertAlmostEqual(metrics["psnr_delta_vs_base"], 0.0, places=6)
            self.assertAlmostEqual(metrics["ssim_delta_vs_base"], 0.0, places=6)
            self.assertAlmostEqual(metrics["l1_improvement_vs_base"], 0.0, places=6)
            self.assertTrue(metrics["remote_sensing_metrics"])
            self.assertIn("panchromatic", metrics["qnr_status"])
            for metric in ("ergas", "sam_degrees", "uiqi", "scc"):
                self.assertIn(metric, metrics)
            per_patch_rows = [
                json.loads(line)
                for line in (output / "per_patch_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(per_patch_rows), 1)
            self.assertEqual(per_patch_rows[0]["dataset_index"], 1)
            self.assertEqual(per_patch_rows[0]["patch"], str(selected_patch))
            self.assertIn("base_psnr", per_patch_rows[0])
            self.assertIn("output_psnr", per_patch_rows[0])
            cache_path = next(output.glob("*_uncertainty.npz"))
            with np.load(cache_path) as cache:
                self.assertIn("base", cache.files)
                self.assertIn("decoder_residual", cache.files)
                self.assertIn("net_addition", cache.files)
                self.assertEqual(cache["base"].shape, (3, 64, 64))
                self.assertEqual(cache["decoder_residual"].shape, (3, 64, 64))
                self.assertEqual(cache["net_addition"].shape, (3, 64, 64))
                self.assertTrue(np.allclose(cache["mean"], cache["base"]))
                self.assertEqual(str(cache["source_patch"]), str(selected_patch))
            log = stream.getvalue()
            self.assertIn("[evaluate] device=cpu", log)
            self.assertIn("selecting val dataset index 1", log)
            self.assertIn("diffusion_unet_passes=1", log)
            self.assertIn("sample 1/1", log)
            self.assertIn("patch 1/1 complete", log)


if __name__ == "__main__":
    unittest.main()
