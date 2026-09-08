import json
from pathlib import Path
import tempfile

import numpy as np
import rasterio

from geodiff_gan.data.manifest import load_manifest
from geodiff_gan.data.oli2msi_128 import _split, find_oli2msi_layout, prepare_oli2msi_128


def _write(path, side, offset):
    values = np.linspace(0, 3000, side * side, dtype=np.float32).reshape(side, side)
    image = np.stack((values + offset, values / 2 + offset, values / 4 + offset))
    with rasterio.open(path, "w", driver="GTiff", width=side, height=side,
                       count=3, dtype="float32") as target:
        target.write(image)


def test_prepare_exact_128_geometry_and_resume():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "OLI2MSI"
        directories = {name: root / name for name in ("train_lr", "train_hr", "test_lr", "test_hr")}
        for directory in directories.values():
            directory.mkdir(parents=True)
        keys = {}
        for wanted in ("train", "val"):
            keys[wanted] = next(f"source_{i}" for i in range(10000) if _split(f"source_{i}", 42, 5) == wanted)
        for key in keys.values():
            _write(directories["train_lr"] / f"{key}.TIF", 160, 10)
            _write(directories["train_hr"] / f"{key}.TIF", 480, 20)
        _write(directories["test_lr"] / "heldout.TIF", 160, 30)
        _write(directories["test_hr"] / "heldout.TIF", 480, 40)

        found, layout = find_oli2msi_layout(directories["train_lr"] / f"{keys['train']}.TIF")
        assert found == root.resolve()
        assert set(layout) == {"train_lr", "train_hr", "test_lr", "test_hr"}
        output = Path(temporary) / "prepared"
        manifest, card = prepare_oli2msi_128(root, output, require_official_counts=False)
        records = load_manifest(manifest, resolve_paths=True)
        assert card["counts"] == {"train": 1, "val": 1, "test": 1}
        for record in records:
            with np.load(record.patch) as data:
                if record.split == "train":
                    assert data["lr"].shape == (3, 160, 160)
                    assert data["hr"].shape == (3, 480, 480)
                else:
                    assert data["lr"].shape == (3, 128, 128)
                    assert data["hr"].shape == (3, 384, 384)
                assert data["lr"].dtype == np.float16
                assert 0 <= data["lr"].min() <= data["lr"].max() <= 1
        _, resumed = prepare_oli2msi_128(root, output, require_official_counts=False)
        assert resumed["status"] == {"skipped": 3}
        assert json.loads((output / "protocol.json").read_text())["model_shape"] == [128, 384]
