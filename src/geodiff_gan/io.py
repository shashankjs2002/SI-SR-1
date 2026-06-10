from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def load_rgb(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            key = "lr" if "lr" in data else "hr"
            array = data[key]
        tensor = torch.from_numpy(array).float()
        if tensor.ndim == 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        return tensor.clamp(0, 1)
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def save_rgb(path: str | Path, tensor: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().cpu().clamp(0, 1)
    if value.ndim == 4:
        value = value[0]
    array = (value.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def save_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

