from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from geodiff_gan.config import load_config
from geodiff_gan.data.manifest import ManifestRecord, write_manifest
from geodiff_gan.training import Trainer


ROOT = Path(__file__).resolve().parents[1]


class TrainingSmokeTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, dict]:
        patch = root / "patch.npz"
        np.savez_compressed(
            patch,
            hr=np.random.default_rng(2).random((3, 64, 64)).astype(np.float32),
        )
        manifest = root / "manifest.jsonl"
        write_manifest(
            manifest,
            [
                ManifestRecord(
                    patch=str(patch),
                    tile_id="TEST_TILE",
                    split="train",
                    row=0,
                    col=0,
                    valid_fraction=1.0,
                    caption="mixed agricultural fields",
                )
            ],
        )
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
            }
        )
        return manifest, config

    def test_base_stage_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = self._fixture(root)
            config["training"].update(
                {
                    "stage": "base",
                    "output_dir": str(root / "run"),
                }
            )
            Trainer(config).train()
            self.assertTrue((root / "run" / "base_epoch_0000.pt").exists())

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


if __name__ == "__main__":
    unittest.main()
