from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..models.degradation import random_degradation
from .manifest import ManifestRecord, load_manifest


class SentinelPatchDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        split: str,
        scale: int = 4,
        caption_file: str | Path | None = None,
        augment: bool = True,
    ) -> None:
        self.records = load_manifest(manifest, split=split)
        self.scale = scale
        self.augment = augment
        self.captions: dict[str, str] = {}
        if caption_file:
            with Path(caption_file).open("r", encoding="utf-8") as handle:
                for line in handle:
                    value = json.loads(line)
                    self.captions[value["patch"]] = value["caption"]

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return image
        if torch.rand(()) < 0.5:
            image = image.flip(-1)
        if torch.rand(()) < 0.5:
            image = image.flip(-2)
        rotations = int(torch.randint(0, 4, ()).item())
        return torch.rot90(image, rotations, dims=(-2, -1))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record: ManifestRecord = self.records[index]
        with np.load(record.patch) as data:
            hr = torch.from_numpy(data["hr"]).float()
        if hr.ndim == 3 and hr.shape[-1] == 3:
            hr = hr.permute(2, 0, 1)
        hr = self._augment(hr.clamp(0, 1))
        lr, degradation = random_degradation(hr.unsqueeze(0), scale=self.scale)
        caption = self.captions.get(record.patch, record.caption)
        return {
            "hr": hr,
            "lr": lr[0],
            "degradation": degradation[0],
            "caption": caption,
            "patch": record.patch,
            "tile_id": record.tile_id,
        }

