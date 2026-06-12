from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from torch import nn


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
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    unwrap(model).load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    return payload
