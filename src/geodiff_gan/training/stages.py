from __future__ import annotations

from torch import nn

STAGES = ("base", "vae", "diffusion", "joint", "prompt_policy", "edit")

STAGE_MODULES: dict[str, tuple[str, ...]] = {
    "base": ("base",),
    "vae": ("vae", "lr_encoder", "mapper", "decoder"),
    "diffusion": ("diffusion",),
    "joint": ("diffusion", "lr_encoder", "mapper", "decoder"),
    "prompt_policy": (),
    "edit": ("diffusion", "mapper", "decoder"),
}


def configure_stage_trainability(model: nn.Module, stage: str) -> tuple[str, ...]:
    if stage not in STAGE_MODULES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")
    model.requires_grad_(False)
    if stage == "prompt_policy":
        mapper = getattr(model, "mapper")
        if mapper.prompt_lr_projection is None:
            return ()
        mapper.prompt_lr_projection.requires_grad_(True)
        mapper.prompt_context_projection.requires_grad_(True)
        mapper.prompt_logit_scale.requires_grad_(True)
        mapper.prompt_logit_bias.requires_grad_(True)
        return (
            "mapper.prompt_lr_projection",
            "mapper.prompt_context_projection",
            "mapper.prompt_logit_scale",
            "mapper.prompt_logit_bias",
        )
    for name in STAGE_MODULES[stage]:
        getattr(model, name).requires_grad_(True)
    return STAGE_MODULES[stage]


def stage_uses_discriminators(stage: str) -> bool:
    if stage not in STAGE_MODULES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")
    return stage in ("joint", "edit")
