from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn

from .checkpoint import unwrap


class ModelEMA:
    """Exponential moving average over a model's complete state."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in unwrap(model).state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = unwrap(model).state_dict()
        if current.keys() != self.shadow.keys():
            raise RuntimeError("EMA state no longer matches the model structure")
        self.updates += 1
        for name, value in current.items():
            shadow = self.shadow[name]
            if torch.is_floating_point(shadow):
                shadow.mul_(self.decay).add_(
                    value.detach().to(shadow.device, dtype=shadow.dtype),
                    alpha=1 - self.decay,
                )
            else:
                shadow.copy_(value.detach().to(shadow.device))

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        target = unwrap(model)
        target.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {name: value.detach().clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        shadow = state.get("shadow")
        if not isinstance(shadow, dict) or shadow.keys() != self.shadow.keys():
            raise ValueError("Invalid or incompatible EMA state")
        self.decay = float(state.get("decay", self.decay))
        self.updates = int(state.get("updates", 0))
        for name, value in shadow.items():
            self.shadow[name].copy_(
                value.to(
                    self.shadow[name].device,
                    dtype=self.shadow[name].dtype,
                )
            )

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        target = unwrap(model)
        backup = {
            name: value.detach().clone()
            for name, value in target.state_dict().items()
        }
        self.copy_to(target)
        try:
            yield
        finally:
            target.load_state_dict(backup, strict=True)
