"""Learning, dispatch, and checkpoint regression tests for residual recovery."""
from dataclasses import replace
import copy
import json

import pytest
import torch

from geodiff_gan.experiments.trust_moe import evaluate, load_model, train
from geodiff_gan.experiments.trust_recovery import (
    audit_checkpoint, curriculum, initialize_residual, recovery_config,
)
from geodiff_gan.models.trust_moe import TrustMoESR, HRConditionedTileExpert
from geodiff_gan.training.trust_losses import tile_risk_target, trust_moe_losses
from test_trust_moe import config_for, fixture_manifest


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


def model_for(**kwargs):
    options = dict(width=8, base_embed_dim=8, base_depth=1, base_groups=1,
                   base_heads=2, router_depth=1, num_experts=2, top_k=1,
                   expert_kind="hr_residual", halo=6, detach_router_features=True)
    options.update(kwargs)
    return TrustMoESR(**options)


def test_hr_conditioned_expert_uses_signed_detail():
    expert = HRConditionedTileExpert(8, 3, 0.1)
    features = torch.rand(1, 8, 8, 8)
    cues = torch.randn(1, 9, 24, 24)
    assert not torch.equal(expert(features, cues), expert(features, -cues))
    assert expert(features, cues).abs().max() <= 0.1


def test_hr_dispatch_geometry_bypass_and_gradients():
    model = model_for().train()
    calls = [0, 0]
    for i, expert in enumerate(model.experts):
        def hook(_, inputs, i=i):
            calls[i] += len(inputs[0])
            assert inputs[1].shape[-1] == 3 * inputs[0].shape[-1]
        expert.register_forward_pre_hook(hook)
    lr = torch.rand(1, 3, 17, 23) * 0.3
    out = model(lr, bypass_trust=True)
    assert out.image.shape == (1, 3, 51, 69)
    assert calls == out.assignments.sum((0, 1)).int().tolist()
    assert torch.all(out.trust == 1)
    skipped = (out.active == 0).expand_as(out.image)
    assert torch.equal(out.image[skipped], out.base[skipped])
    loss, _ = trust_moe_losses(out, torch.rand_like(out.image) * 0.3,
                               torch.ones_like(out.active), 8, 3,
                               dict(proposal_mse=100, mse=100, risk=.02, balance=.01),
                               False, "gain")
    loss.backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.router.risk.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())
    assert not any(isinstance(m, torch.nn.PixelShuffle) for m in model.modules())
    assert torch.equal(model(lr, residual_scale=0).image, model(lr, base_only=True).image)


def test_gain_labels_mask_missing_proposals_and_reward_repair():
    model = model_for(coverage=1.0)
    out = model(torch.full((1, 3, 16, 16), 0.3))
    residual = torch.full_like(out.image, 0.03)
    mask = torch.ones_like(out.active)
    out = replace(out, residual=residual, active=mask)
    good, valid = tile_risk_target(out, out.base + residual, mask, 8, 3, "gain")
    assert valid.all() and torch.all(good > .99)
    bad, _ = tile_risk_target(out, out.base - residual, mask, 8, 3, "gain")
    assert torch.all(bad < -.99)
    missing = mask.clone()
    missing[:, :, :24, :24] = 0
    _, observed = tile_risk_target(replace(out, active=missing), out.base + residual, mask, 8, 3, "gain")
    assert observed.tolist() == [[False, True, True, True]]
    _, empty = tile_risk_target(out, out.base + residual, mask * 0, 8, 3, "gain")
    assert not empty.any()


def test_auxiliary_risk_does_not_override_reconstruction_encoder():
    model = model_for(coverage=1.0)
    out = model(torch.rand(1, 3, 16, 16))
    out.difficulty.sum().backward()
    assert model.encoder[0].weight.grad is None
    assert model.router.risk.weight.grad is not None


def test_expert_can_learn_nonzero_correction():
    model = model_for(num_experts=1, coverage=1, use_trust=False)
    lr = torch.rand(1, 3, 8, 8) * .2 + .2
    with torch.no_grad():
        base = model(lr, base_only=True).base
        # A learnable correction tied to the observed base, not random HR noise.
        target = base + .015 + .02 * (base - base.mean())
        initial_error = (base - target).square().mean()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.003)
    for _ in range(60):
        optimizer.zero_grad()
        out = model(lr)
        loss, _ = trust_moe_losses(out, target, torch.ones_like(out.active), 8, 3,
                                   dict(mse=100, proposal_mse=100), False, "gain")
        loss.backward()
        optimizer.step()
    out = model(lr)
    assert (out.image - target).square().mean() < initial_error * .2
    assert (out.image - base).abs().mean() > .005
    assert all(p.grad is None for p in model.base.parameters())


def test_curriculum_delays_guard_and_uses_deployed_policy():
    config = recovery_config("manifest", "run", parent="base", profile="sparse_transformer")
    weights, opts, warm = curriculum(config, 0)
    assert warm and opts == dict(coverage=1.0, bypass_trust=True)
    assert weights["guard"] == weights["trust"] == 0
    weights, opts, warm = curriculum(config, 4)
    assert not warm and not opts and weights["guard"] == config["losses"]["guard"]


def test_recovery_checkpoint_transfer_resume_and_audit(tmp_path):
    manifest = fixture_manifest(tmp_path, size=32)
    parent = train(config_for(manifest, tmp_path / "base"), "cpu")
    dimensions = dict(width=8, base_embed_dim=8, base_depth=1, base_groups=1,
                       base_heads=2, router_depth=1)
    config = recovery_config(manifest, tmp_path / "single", parent=parent, epochs=1,
                             batch_size=1, crop_size=16, experts=2, top_k=1,
                             dataset_id="fixture", base_model=dimensions)
    config["training"].update(num_workers=0, amp=False)
    initial = train(config, "cpu")
    multi = copy.deepcopy(config)
    multi.update(root=str(tmp_path / "multi"), profile="sparse_transformer", residual_initializer=str(initial))
    multi["model"].update(num_experts=2, coverage=.5, use_trust=True)
    multi["training"]["residual_warmup_epochs"] = 1
    with pytest.raises(RuntimeError, match="Only warm-up"):
        train(multi, "cpu")
    assert not (tmp_path / "multi/best.pt").exists()
    multi["training"]["epochs"] = 2
    trained = train(multi, "cpu")
    saved = torch.load(tmp_path / "multi/last.pt", weights_only=False)
    assert saved["epoch"] == 2 and saved["optimizer_steps"] == 2
    assert saved["lineage"]["initializer_sha256"]
    assert saved["lineage"]["parent_sha256"]
    model, _ = load_model(trained)
    before = {k: v.clone() for k, v in model.base.state_dict().items()}
    initialize_residual(model, initial, dataset_id="fixture")
    assert all(torch.equal(v, model.base.state_dict()[k]) for k, v in before.items())
    with torch.no_grad():
        next(model.base.parameters()).add_(1)
    with pytest.raises(ValueError, match="different base"):
        initialize_residual(model, initial, dataset_id="fixture")
    report = audit_checkpoint(trained, manifest, tmp_path / "audit.json", device="cpu")
    assert report["base_frozen"]
    assert report["reconstruction_gradients"]["encoder"] > 0
    result = evaluate(trained, tmp_path / "evaluation", "val", device="cpu")
    assert result["correction_abs_mean"] > 0
    history = json.loads((tmp_path / "multi/history/epoch_0002.json").read_text())
    assert history["first_batch_gradient_norms"]["encoder"] > 0


def test_memorization_model_cannot_enter_final_evaluation(tmp_path):
    manifest = fixture_manifest(tmp_path, size=32)
    parent = train(config_for(manifest, tmp_path / "base"), "cpu")
    config = config_for(manifest, tmp_path / "diagnostic", "single_expert", parent)
    config["training"]["diagnostic_overfit_pairs"] = 1
    checkpoint = train(config, "cpu")
    with pytest.raises(ValueError, match="Memorization"):
        evaluate(checkpoint, tmp_path / "test", "test", device="cpu")


def test_new_notebook_executes_training_through_download(tmp_path, monkeypatch):
    """Execute real notebook cells on CPU pairs, including all primary controls."""
    import ast
    import os
    import shutil
    import sys
    import time
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from geodiff_gan.experiments import trust_moe as workflow, trust_report as reporting
    from geodiff_gan.experiments import trust_recovery as recovery
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "kaggle/GeoDiff_TrustMoE_OLI2MSI_Residual_Recovery_3x.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    manifest = fixture_manifest(tmp_path)
    suite = tmp_path / "suite"
    for name in ("figures", "reports", "diagnostics"):
        (suite / name).mkdir(parents=True)

    def run_command(command, cwd=None):
        cfg = json.loads(Path(command[-1]).read_text())
        cfg["training"].update(num_workers=0, amp=False)
        return workflow.train(cfg, "cpu")

    def quick_timing(*args, **kwargs):
        return reporting.benchmark(*args, **kwargs, repeats=1, warmup=0, frames=1, device="cpu")

    scope = dict(Path=Path, json=json, os=os, shutil=shutil, sys=sys, time=time,
        torch=torch, np=np, pd=pd, plt=plt, REPOSITORY_DIR=root, SUITE_ROOT=suite,
        OLD_SUITE_ROOT=tmp_path / "missing", BASE_CHECKPOINTS={42: None},
        SEEDS=[42], EPOCHS=dict(base=1, diagnostic=1, single=1, residual=3),
        BATCH_SIZE=1, TRAIN_LR_CROP=16, NUM_EXPERTS=2, TOP_K=1, REGION_FRACTION=.5,
        BASE_MODEL=dict(width=8, base_embed_dim=8, base_depth=1, base_groups=1,
                        base_heads=2, router_depth=1, tile_size=8),
        AUDIT={"dataset_id": "fixture"}, MANIFEST=manifest, DATA_CARD={"protocol": "fixture"},
        DIAGNOSTIC_PAIRS=1, REQUIRE_RECOVERY_SCREEN=False, FAST_DEV_RUN=False,
        RUN_TEST_EVALUATION=True, RUN=dict(dense_gain=True, sparse_error=True, sparse_gain=True,
            sparse_no_trust=True, sparse_uniform=True, sparse_conv=False,
            sparse_adversarial=False, adaptive_k=False, legacy_expert_control=False),
        display=lambda *args: None, FileLink=lambda *args, **kwargs: None,
        run=run_command, benchmark=quick_timing,
        **{name: getattr(workflow, name) for name in (
            "make_config", "write_json", "adopt_base", "digest_file", "evaluate", "dataset_for")},
        **{name: getattr(recovery, name) for name in (
            "recovery_config", "audit_checkpoint", "memorization_report", "validation_screen")},
        **{name: getattr(reporting, name) for name in (
            "METRICS", "summarize", "paired_intervals", "compare_controls", "bundle_results")})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    for cell in notebook["cells"][9:]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = source.replace("scores['count'] != 100", "scores['count'] != 1")
        source = source.replace("/kaggle/working", tmp_path.as_posix())
        exec(compile(source, "<recovery-notebook>", "exec"), scope)
    assert len(scope["TEST_EVALS"]) == 7
    assert (suite / "reports/test/controls_42.csv").exists()
    assert (suite / "figures/val_42_000000_corrections.png").exists()
    assert (suite / "example_inputs/test_000000_hr.tif").exists()
    assert scope["archive"].exists()
    # Indexed visualization defaults to test after the run, including LR and HR.
    scope["show_result"](0)
    assert (suite / "figures/test_42_000000.png").exists()
