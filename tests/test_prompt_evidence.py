from __future__ import annotations

import torch

from geodiff_gan.config import load_config
from geodiff_gan.losses import (
    prompt_contradiction_loss,
    prompt_evidence_policy_loss,
    prompt_utility_loss,
)
from geodiff_gan.models.system import GeoDiffGAN
from geodiff_gan.text import (
    HashTextEncoder,
    controlled_prompt_variants,
    mismatched_prompt,
)
from geodiff_gan.training.stages import configure_stage_trainability


def _config() -> dict:
    config = load_config("configs/smoke.yaml", "configs/default.yaml")
    config["model"]["use_prompt_evidence_controller"] = True
    return config


def test_controlled_prompt_variants_are_distinct_and_deterministic() -> None:
    prompt = "dense urban blocks with intersecting roads"
    first = controlled_prompt_variants(prompt)
    second = controlled_prompt_variants(prompt)
    assert first == second
    assert first["matched"] == prompt
    assert first["null"] == ""
    assert first["paraphrase"] != prompt
    assert first["mismatch"] != prompt
    assert mismatched_prompt(prompt) == first["mismatch"]


def test_prompt_controller_returns_bounded_support_and_permission() -> None:
    torch.manual_seed(11)
    config = _config()
    model = GeoDiffGAN.from_config(config).eval()
    lr = torch.rand(2, 3, 16, 16)
    context = HashTextEncoder(32, 8)(
        ["urban roads and roofs", "agricultural fields"]
    )
    degradation = torch.rand(2, 4)
    with torch.no_grad():
        base = model.base(lr)
        latent = torch.randn(2, 4, 8, 8)
        output = model.decode_latent(
            latent,
            lr,
            context,
            degradation,
            mode="sr",
            base=base,
            back_projection_steps=0,
        )
    assert output.prompt_support.shape == (2, 1, 8, 8)
    assert output.prompt_permission.shape == (2, 1, 8, 8)
    assert torch.all((0 <= output.prompt_support) & (output.prompt_support <= 1))
    assert torch.all(
        (0 <= output.prompt_permission) & (output.prompt_permission <= 1)
    )
    assert torch.all(output.prompt_permission <= output.prompt_support + 1e-6)
    assert output.metadata[0]["prompt_evidence_controller"] is True


def test_prompt_policy_stage_trains_only_controller_parameters() -> None:
    model = GeoDiffGAN.from_config(_config())
    configured = configure_stage_trainability(model, "prompt_policy")
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert set(configured) == {
        "mapper.prompt_lr_projection",
        "mapper.prompt_context_projection",
        "mapper.prompt_logit_scale",
        "mapper.prompt_logit_bias",
    }
    assert trainable
    assert all(name.startswith("mapper.prompt_") for name in trainable)


def test_prompt_policy_loss_rewards_supported_and_rejected_examples() -> None:
    kinds = ["original", "paraphrase", "null", "mismatch"]
    correct_support = torch.tensor([0.95, 0.9, 0.05, 0.1]).view(4, 1, 1, 1)
    correct_permission = torch.tensor([0.4, 0.3, 0.01, 0.02]).view(4, 1, 1, 1)
    wrong_support = 1.0 - correct_support
    wrong_permission = torch.full_like(correct_permission, 0.5)
    correct = prompt_evidence_policy_loss(
        correct_support,
        correct_permission,
        kinds,
    )
    wrong = prompt_evidence_policy_loss(
        wrong_support,
        wrong_permission,
        kinds,
    )
    assert correct < wrong


def test_contradiction_and_utility_losses_use_correct_prompt_subsets() -> None:
    null = torch.zeros(3, 1, 2, 2)
    prompted = null.clone()
    prompted[0] = 0.2
    prompted[1] = 0.3
    prompted[2] = 0.4
    kinds = ["original", "mismatch", "null"]
    contradiction = prompt_contradiction_loss(prompted, null, kinds)
    assert torch.isclose(contradiction, torch.tensor(0.3))

    target = torch.zeros_like(prompted)
    utility = prompt_utility_loss(prompted, null, target, kinds)
    assert torch.isclose(utility, torch.tensor(0.2))
