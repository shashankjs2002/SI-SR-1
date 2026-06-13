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
            np.savez_compressed(
                patch,
                hr=np.random.default_rng(5).random((3, 64, 64)).astype(np.float32),
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
                    )
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
                "--limit",
                "1",
                "--device",
                "cpu",
                "--progress",
                "compact",
                "--no-text",
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
            log = stream.getvalue()
            self.assertIn("[evaluate] device=cpu", log)
            self.assertIn("diffusion_unet_passes=1", log)
            self.assertIn("sample 1/1", log)
            self.assertIn("patch 1/1 complete", log)


if __name__ == "__main__":
    unittest.main()
