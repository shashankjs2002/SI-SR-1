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
            "positional",
        ),
        augment: bool = True,
        random_degradation: bool | None = None,
        degradation_seed: int = 0,
        degradation_severity: str = "mild",
        target_key: str = "hr",
        condition_key: str | None = None,
        radiometric_calibration: str | Path | None = None,
        output_channels: int = 3,
        input_mode: str = "synthetic",
    ) -> None:
        self.records = load_manifest(manifest, split=split)
        self.scale = scale
        self.augment = augment
        self.random_degradation = augment if random_degradation is None else random_degradation
        self.degradation_seed = degradation_seed
        self.degradation_severity = degradation_severity
        self.target_key = target_key
        self.condition_key = condition_key
        self.radiometric_slope: torch.Tensor | None = None
        self.radiometric_offset: torch.Tensor | None = None
        if radiometric_calibration:
            calibration_path = Path(radiometric_calibration)
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            slope = calibration.get("slope")
            offset = calibration.get("offset")
            if not isinstance(slope, list) or not isinstance(offset, list):
                raise ValueError(
                    f"Invalid radiometric calibration: {calibration_path}"
                )
            if not slope or len(slope) != len(offset):
                raise ValueError(
                    "Radiometric slope and offset must be equally sized non-empty lists"
                )
            self.radiometric_slope = torch.tensor(slope, dtype=torch.float32)
            self.radiometric_offset = torch.tensor(offset, dtype=torch.float32)
        self.output_channels = output_channels
        if input_mode not in ("synthetic", "paired"):
            raise ValueError("input_mode must be 'synthetic' or 'paired'")
        self.input_mode = input_mode
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

    @staticmethod
    def _to_channel_first(array: np.ndarray, key: str) -> torch.Tensor:
        tensor = torch.from_numpy(array).float()
        if tensor.ndim != 3:
            raise ValueError(f"Patch key {key!r} must be a 3D tensor")
        if tensor.shape[0] <= 16 and tensor.shape[1] > 16 and tensor.shape[2] > 16:
            return tensor
        if tensor.shape[-1] <= 16 and tensor.shape[0] > 16 and tensor.shape[1] > 16:
            return tensor.permute(2, 0, 1)
        raise ValueError(
            f"Patch key {key!r} has ambiguous shape {tuple(tensor.shape)}; "
            "expected CHW or HWC with a small channel dimension"
        )

    def _augmentation_parameters(self) -> tuple[bool, bool, int]:
        if not self.augment:
            return False, False, 0
        return (
            bool(torch.rand(()) < 0.5),
            bool(torch.rand(()) < 0.5),
            int(torch.randint(0, 4, ()).item()),
        )

    @staticmethod
    def _apply_augment(
        image: torch.Tensor, flip_width: bool, flip_height: bool, rotations: int
    ) -> torch.Tensor:
        if flip_width:
            image = image.flip(-1)
        if flip_height:
            image = image.flip(-2)
        return torch.rot90(image, rotations, dims=(-2, -1))

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        return self._apply_augment(image, *self._augmentation_parameters())

    def _apply_radiometric_calibration(
        self, image: torch.Tensor
    ) -> torch.Tensor:
        if self.radiometric_slope is None or self.radiometric_offset is None:
            return image
        channels = self.radiometric_slope.numel()
        if image.shape[0] < channels:
            raise ValueError(
                f"Radiometric calibration requires {channels} channels, "
                f"but input has {image.shape[0]}"
            )
        calibrated = image.clone()
        calibrated[:channels] = (
            calibrated[:channels]
            * self.radiometric_slope[:, None, None]
            + self.radiometric_offset[:, None, None]
        ).clamp(0, 1)
        return calibrated

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record: ManifestRecord = self.records[index]
        with np.load(record.patch) as data:
            if self.target_key not in data.files:
                raise KeyError(
                    f"Patch {record.patch} does not contain target key "
                    f"{self.target_key!r}"
                )
            target = self._to_channel_first(data[self.target_key], self.target_key)
            if self.input_mode == "paired":
                paired_lr_key = self.condition_key or "lr"
                if paired_lr_key not in data.files:
                    raise KeyError(
                        f"Paired patch {record.patch} does not contain "
                        f"{paired_lr_key!r}"
                    )
                clean_lr_key = (
                    "clean_lr_ms" if paired_lr_key == "lr_ms" else "clean_lr"
                )
                stored_lr = self._to_channel_first(
                    data[paired_lr_key], paired_lr_key
                )
                stored_clean_lr = self._to_channel_first(
                    data[clean_lr_key]
                    if clean_lr_key in data.files
                    else data[paired_lr_key],
                    clean_lr_key,
                )
                stored_raw_lr = stored_lr
                stored_lr = self._apply_radiometric_calibration(stored_lr)
                stored_clean_lr = self._apply_radiometric_calibration(
                    stored_clean_lr
                )
                stored_degradation = torch.from_numpy(
                    data["degradation"]
                    if "degradation" in data.files
                    else np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
                ).float()
                valid_mask = self._to_channel_first(
                    data["valid_mask_hr"]
                    if "valid_mask_hr" in data.files
                    else np.ones((1, *target.shape[-2:]), dtype=np.float32),
                    "valid_mask_hr",
                )
                stored_valid_mask_lr = self._to_channel_first(
                    data["valid_mask_lr"]
                    if "valid_mask_lr" in data.files
                    else np.ones((1, *stored_lr.shape[-2:]), dtype=np.float32),
                    "valid_mask_lr",
                )
                condition = target
            else:
                condition_key = self.condition_key or self.target_key
                if condition_key not in data.files:
                    raise KeyError(
                        f"Patch {record.patch} does not contain condition key "
                        f"{condition_key!r}. Re-run preprocessing with multispectral "
                        "bands or switch data.condition_key back to null."
                    )
                condition = self._to_channel_first(data[condition_key], condition_key)
        if self.input_mode == "synthetic" and target.shape[-2:] != condition.shape[-2:]:
            raise ValueError(
                f"Target and condition spatial shapes differ for {record.patch}: "
                f"{tuple(target.shape[-2:])} vs {tuple(condition.shape[-2:])}"
            )
        if target.shape[0] < self.output_channels:
            raise ValueError(
                f"Patch {record.patch} has {target.shape[0]} target channels, "
                f"but {self.output_channels} output channels were requested"
            )
        flip_width, flip_height, rotations = self._augmentation_parameters()
        target = self._apply_augment(target.clamp(0, 1), flip_width, flip_height, rotations)
        condition = self._apply_augment(
            condition.clamp(0, 1), flip_width, flip_height, rotations
        )
        hr = target[: self.output_channels]
        if self.input_mode == "paired":
            lr_value = self._apply_augment(
                stored_lr.clamp(0, 1), flip_width, flip_height, rotations
            )
            raw_lr_value = self._apply_augment(
                stored_raw_lr.clamp(0, 1), flip_width, flip_height, rotations
            )
            clean_lr_value = self._apply_augment(
                stored_clean_lr.clamp(0, 1), flip_width, flip_height, rotations
            )
            valid_mask = self._apply_augment(
                valid_mask.float(), flip_width, flip_height, rotations
            )
            valid_mask_lr = self._apply_augment(
                stored_valid_mask_lr.float(), flip_width, flip_height, rotations
            )
            expected_hr = (lr_value.shape[-2] * self.scale, lr_value.shape[-1] * self.scale)
            if tuple(hr.shape[-2:]) != expected_hr:
                raise ValueError(
                    f"Paired patch {record.patch} violates {self.scale}x geometry: "
                    f"LR={tuple(lr_value.shape[-2:])}, HR={tuple(hr.shape[-2:])}"
                )
            if tuple(valid_mask.shape[-2:]) != tuple(hr.shape[-2:]):
                raise ValueError(
                    f"Paired patch {record.patch} has an HR validity mask with "
                    f"shape {tuple(valid_mask.shape[-2:])}, expected "
                    f"{tuple(hr.shape[-2:])}"
                )
            if tuple(valid_mask_lr.shape[-2:]) != tuple(lr_value.shape[-2:]):
                raise ValueError(
                    f"Paired patch {record.patch} has an LR validity mask with "
                    f"shape {tuple(valid_mask_lr.shape[-2:])}, expected "
                    f"{tuple(lr_value.shape[-2:])}"
                )
            lr = lr_value.unsqueeze(0)
            clean_lr = clean_lr_value.unsqueeze(0)
            degradation = stored_degradation.reshape(1, -1)
        else:
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
                condition.unsqueeze(0),
                scale=self.scale,
                generator=generator,
                return_clean=True,
                severity=self.degradation_severity,
            )
            valid_mask = torch.ones((1, *hr.shape[-2:]), dtype=hr.dtype)
            valid_mask_lr = torch.ones(
                (1, *lr.shape[-2:]), dtype=hr.dtype
            )
            raw_lr_value = lr[0]
        lr_rgb = lr[:, : self.output_channels]
        raw_lr_rgb = raw_lr_value.unsqueeze(0)[:, : self.output_channels]
        clean_lr_rgb = clean_lr[:, : self.output_channels]
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
            "lr_rgb": lr_rgb[0],
            "lr_raw_rgb": raw_lr_rgb[0],
            "clean_lr": clean_lr_rgb[0],
            "degradation": degradation[0],
            "valid_mask": valid_mask,
            "valid_mask_lr": valid_mask_lr,
            "caption": caption,
            "patch": record.patch,
            "tile_id": record.tile_id,
        }
