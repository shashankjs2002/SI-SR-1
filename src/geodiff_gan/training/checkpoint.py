from __future__ import annotations

import re
import shutil
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn


def checkpoint_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a checkpoint so downstream runs can verify their parent lineage."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def latest_stage_checkpoint(
    output_dir: str | Path,
    stage: str,
) -> Path | None:
    directory = Path(output_dir)
    pattern = re.compile(rf"^{re.escape(stage)}_epoch_(\d+)\.pt$")
    candidates = []
    for path in directory.glob(f"{stage}_epoch_*.pt"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def best_stage_checkpoint(
    output_dir: str | Path,
    stage: str,
) -> Path | None:
    path = Path(output_dir) / f"{stage}_best.pt"
    return path if path.exists() else None


def copy_checkpoint(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    shutil.copy2(source_path, temporary)
    temporary.replace(destination_path)


def prune_stage_epoch_checkpoints(
    output_dir: str | Path,
    stage: str,
    keep: str | Path,
) -> None:
    directory = Path(output_dir)
    keep_path = Path(keep).resolve()
    pattern = re.compile(rf"^{re.escape(stage)}_epoch_(\d+)\.pt$")
    for path in directory.glob(f"{stage}_epoch_*.pt"):
        if pattern.match(path.name) and path.resolve() != keep_path:
            path.unlink()


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    stage: str,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "stage": stage,
        "config": config,
        "extra": extra or {},
    }
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
    prefer_ema: bool = False,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_state = payload["model"]
    if prefer_ema:
        ema_state = payload.get("extra", {}).get("ema")
        if isinstance(ema_state, dict) and isinstance(ema_state.get("shadow"), dict):
            model_state = ema_state["shadow"]
    unwrap(model).load_state_dict(model_state, strict=strict)
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    return payload
