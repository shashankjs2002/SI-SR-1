from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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
        caption_field: str = "caption",
        caption_sampling: str = "fixed",
        random_caption_fields: tuple[str, ...] | list[str] = (
            "brief",
            "descriptive",
            "analytical",
        ),
        augment: bool = True,
        random_degradation: bool | None = None,
        degradation_seed: int = 0,
        degradation_severity: str = "mild",
    ) -> None:
        self.records = load_manifest(manifest, split=split)
        self.scale = scale
        self.augment = augment
        self.random_degradation = augment if random_degradation is None else random_degradation
        self.degradation_seed = degradation_seed
        self.degradation_severity = degradation_severity
        self.caption_field = caption_field
        if caption_sampling not in ("fixed", "random"):
            raise ValueError("caption_sampling must be 'fixed' or 'random'")
        self.caption_sampling = caption_sampling
        self.random_caption_fields = tuple(random_caption_fields)
        self.captions: dict[str, str] = {}
        self.caption_variants: dict[str, dict[str, str]] = {}
        if caption_file:
            with Path(caption_file).open("r", encoding="utf-8") as handle:
                for line in handle:
                    value = json.loads(line)
                    patch = str(value["patch"])
                    caption = self._caption_from_record(value)
                    for key in self._caption_keys(patch):
                        self.captions.setdefault(key, caption)
                    variants = self._caption_variants_from_record(value)
                    if variants:
                        for key in self._caption_keys(patch):
                            self.caption_variants.setdefault(key, variants)

    @staticmethod
    def _caption_keys(patch: str) -> tuple[str, ...]:
        path = Path(patch)
        keys = [
            patch,
            path.as_posix(),
            path.name,
        ]
        if len(path.parts) >= 2:
            keys.append(Path(*path.parts[-2:]).as_posix())
        if len(path.parts) >= 3:
            keys.append(Path(*path.parts[-3:]).as_posix())
        return tuple(dict.fromkeys(keys))

    def _caption_from_record(self, value: dict[str, Any]) -> str:
        captions = value.get("captions")
        if isinstance(captions, dict):
            selected = captions.get(self.caption_field)
            if selected is None and self.caption_field == "caption":
                selected = captions.get("descriptive")
            if isinstance(selected, str):
                return selected
            if isinstance(selected, dict):
                return json.dumps(selected, ensure_ascii=True, sort_keys=True)
        selected = value.get(self.caption_field)
        if isinstance(selected, str):
            return selected
        if isinstance(selected, dict):
            return json.dumps(selected, ensure_ascii=True, sort_keys=True)
        selected = value.get("caption", "")
        if isinstance(selected, str):
            return selected
        return json.dumps(selected, ensure_ascii=True, sort_keys=True)

    @staticmethod
    def _stringify_caption_variant(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=True, sort_keys=True)
        if isinstance(value, list):
            return ", ".join(map(str, value))
        return str(value)

    def _caption_variants_from_record(self, value: dict[str, Any]) -> dict[str, str]:
        variants: dict[str, str] = {}
        captions = value.get("captions")
        if isinstance(captions, dict):
            for key in self.random_caption_fields:
                if key in captions:
                    variants[key] = self._stringify_caption_variant(captions[key])
        for key in self.random_caption_fields:
            if key in value and key not in variants:
                variants[key] = self._stringify_caption_variant(value[key])
        if "caption" in value and "caption" not in variants:
            variants["caption"] = self._stringify_caption_variant(value["caption"])
        return {key: caption for key, caption in variants.items() if caption.strip()}

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
        generator = None
        if not self.random_degradation:
            key = (
                f"{self.degradation_seed}:{record.tile_id}:"
                f"{record.row}:{record.col}:{record.patch}"
            )
            seed = int.from_bytes(
                hashlib.sha256(key.encode("utf-8")).digest()[:8],
                byteorder="little",
                signed=False,
            )
            generator = torch.Generator().manual_seed(seed)
        lr, degradation, clean_lr = random_degradation(
            hr.unsqueeze(0),
            scale=self.scale,
            generator=generator,
            return_clean=True,
            severity=self.degradation_severity,
        )
        caption = record.caption
        variants: dict[str, str] | None = None
        for key in self._caption_keys(record.patch):
            if key in self.captions:
                caption = self.captions[key]
            if key in self.caption_variants:
                variants = self.caption_variants[key]
                break
        if self.caption_sampling == "random" and variants:
            available = [
                variants[field]
                for field in self.random_caption_fields
                if field in variants and variants[field].strip()
            ]
            if not available and "caption" in variants:
                available = [variants["caption"]]
            if available:
                caption = available[int(torch.randint(0, len(available), ()).item())]
        return {
            "hr": hr,
            "lr": lr[0],
            "clean_lr": clean_lr[0],
            "degradation": degradation[0],
            "caption": caption,
            "patch": record.patch,
            "tile_id": record.tile_id,
        }
