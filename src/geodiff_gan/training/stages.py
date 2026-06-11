from __future__ import annotations

from torch import nn

STAGES = ("base", "vae", "diffusion", "joint", "edit")

STAGE_MODULES: dict[str, tuple[str, ...]] = {
    "base": ("base",),
    "vae": ("vae", "lr_encoder", "mapper", "decoder"),
    "diffusion": ("diffusion",),
    "joint": ("diffusion", "lr_encoder", "mapper", "decoder"),
    "edit": ("diffusion", "mapper", "decoder"),
}


def configure_stage_trainability(model: nn.Module, stage: str) -> tuple[str, ...]:
    if stage not in STAGE_MODULES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")
    model.requires_grad_(False)
    for name in STAGE_MODULES[stage]:
        getattr(model, name).requires_grad_(True)
    return STAGE_MODULES[stage]


def stage_uses_discriminators(stage: str) -> bool:
    if stage not in STAGE_MODULES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")
    return stage in ("joint", "edit")
