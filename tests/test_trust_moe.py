from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from geodiff_gan.data.manifest import ManifestRecord, write_manifest
from geodiff_gan.experiments.trust_moe import (
    adopt_base, dataset_for, evaluate, load_model, make_config, prepare_manifest, train,
)
from geodiff_gan.experiments.trust_report import benchmark, paired_intervals
from geodiff_gan.models.trust_moe import TrustMoESR
from geodiff_gan.training.trust_losses import local_trust_target, trust_moe_losses


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(**kwargs):
    return TrustMoESR(width=8, base_embed_dim=8, base_heads=2, base_depth=1,
                      base_groups=1, router_depth=1, num_experts=3, **kwargs)


def fixture_manifest(root, size=32):
    records = []
    for i, split in enumerate(("train", "val", "test")):
        lr = np.random.default_rng(i).random((3, size, size), dtype=np.float32) * 0.2
        hr = np.repeat(np.repeat(lr, 3, 1), 3, 2)
        path = root / f"{split}.npz"
        np.savez(path, lr=lr, hr=hr, valid_mask_lr=np.ones((1, size, size), np.float32),
                 valid_mask_hr=np.ones((1, size * 3, size * 3), np.float32))
        records.append(ManifestRecord(str(path), "tile", split, 0, i * size * 6, 1.0,
                                      scale=3, scene_class="urban"))
    path = root / "manifest.jsonl"
    write_manifest(path, records)
    return path


def config_for(manifest, root, profile="base", parent=None):
    config = make_config(manifest, root, profile=profile, parent=parent, epochs=1,
                         batch_size=1, crop_size=16, experts=3, top_k=2, dataset_id="fixture")
    config["model"].update(width=8, base_embed_dim=8, base_heads=2, base_depth=1,
                           base_groups=1, router_depth=1)
    config["training"].update(num_workers=0, amp=False, ema_decay=0.9)
    return config


@pytest.mark.parametrize("shape", [(16, 24), (17, 23)])
def test_geometry_and_actual_dispatch(shape):
    model = tiny_model().eval()
    counts = [0] * model.num_experts
    handles = []
    for k, expert in enumerate(model.experts):
        def hook(module, inputs, k=k):
            counts[k] += len(inputs[0])
        handles.append(expert.register_forward_pre_hook(hook))
    lr = torch.rand(2, 3, *shape) * 0.2
    output = model(lr)
    assert output.image.shape == (2, 3, shape[0] * 3, shape[1] * 3)
    assert counts == output.assignments.sum((0, 1)).int().tolist()
    assert sum(counts) == int(output.dispatched_tiles)
    assert 0 < sum(counts) < output.assignments.shape[0] * output.assignments.shape[1] * 2
    skipped = (output.active == 0).expand_as(output.image)
    assert torch.equal(output.image[skipped], output.base[skipped])
    assert output.trust.min() >= 0 and output.trust.max() <= 1
    assert not any(isinstance(m, torch.nn.PixelShuffle) for m in model.modules())
    assert not any(name in dict(model.named_children()) for name in ("diffusion", "vae", "mapper"))
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("option", [{"coverage": 0}, {"residual_scale": 0}, {"base_only": True}])
def test_exact_base_and_zero_expert_calls(option):
    model = tiny_model().eval()
    def fail(*args):
        raise AssertionError("Expert should not execute")
    for expert in model.experts:
        expert.register_forward_pre_hook(fail)
    output = model(torch.rand(1, 3, 16, 16), **option)
    assert torch.equal(output.image, output.base)
    assert output.dispatched_tiles == 0


@pytest.mark.parametrize("top_k", [1, 2])
def test_router_gradients_and_frozen_base(top_k):
    model = tiny_model(top_k=top_k).train()
    output = model(torch.rand(2, 3, 16, 16) * 0.2)
    weights = dict(mse=100, proposal=1, trust=0.01, risk=0.01, balance=0.01)
    loss, _ = trust_moe_losses(output, torch.rand_like(output.image) * 0.2,
                               torch.ones_like(output.active), 8, 3, weights)
    loss.backward()
    for module in (model.router.routes, model.router.risk, model.encoder, model.trust_head):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert all(p.grad is None for p in model.base.parameters())


def test_adaptive_k_varies_within_active_regions():
    model = tiny_model(adaptive_k=True).eval()
    result = model(torch.rand(1, 3, 32, 32), coverage=0.5)
    slots = result.assignments.sum(-1)
    assert set(slots[slots > 0].tolist()) == {1, 2}
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, 16, 16), top_k=4)


def test_per_band_trust_oracle():
    residual = torch.full((1, 3, 12, 12), 0.1)
    target = residual.clone()
    target[:, 1] *= -1
    target[:, 2] *= 0.5
    mask = torch.ones(1, 1, 12, 12)
    coefficient = local_trust_target(residual, target, mask)
    assert torch.allclose(coefficient[:, 0], torch.ones_like(coefficient[:, 0]), atol=1e-4)
    assert torch.equal(coefficient[:, 1], torch.zeros_like(coefficient[:, 1]))
    assert torch.allclose(coefficient[:, 2], torch.full_like(coefficient[:, 2], 0.5), atol=1e-4)
    assert local_trust_target(residual * 0, target, mask).count_nonzero() == 0


def test_six_band_geometry_and_trust():
    model = tiny_model(input_channels=6, output_channels=6).eval()
    result = model(torch.rand(1, 6, 16, 16))
    assert result.image.shape == (1, 6, 48, 48)
    assert result.base.shape == result.residual.shape == result.trust.shape == result.image.shape
    assert model.base_input_channels == model.output_channels == 6


def test_six_band_training_and_index_evaluation(tmp_path):
    records = []
    for index, split in enumerate(("train", "val", "test")):
        lr_ms = np.random.default_rng(index + 50).random((6, 24, 24), dtype=np.float32) * 0.8 + 0.1
        hr_ms = np.repeat(np.repeat(lr_ms, 3, 1), 3, 2)
        path = tmp_path / f"{split}_ms.npz"
        np.savez_compressed(
            path, lr=lr_ms[:3], hr=hr_ms[:3], lr_ms=lr_ms, hr_ms=hr_ms,
            clean_lr=lr_ms[:3], clean_lr_ms=lr_ms,
            valid_mask_lr=np.ones((1, 24, 24), np.float32),
            valid_mask_hr=np.ones((1, 72, 72), np.float32),
        )
        records.append(ManifestRecord(str(path), "tile", split, 0, index * 100,
                                      1.0, scale=3, scene_class="mixed"))
    manifest = tmp_path / "manifest_ms.jsonl"
    write_manifest(manifest, records)
    config = make_config(
        manifest, tmp_path / "base_ms", profile="base", epochs=1,
        batch_size=1, crop_size=24, dataset_id="ms-fixture",
        input_channels=6, output_channels=6, condition_key="lr_ms",
        target_key="hr_ms",
        band_names=("red", "green", "blue", "nir", "swir1", "swir2"),
    )
    config["model"].update(width=8, base_embed_dim=8, base_heads=2,
                           base_depth=1, base_groups=1, router_depth=1)
    config["training"].update(num_workers=0, amp=False, ema_decay=0.9)
    checkpoint = train(config, "cpu")
    result = evaluate(checkpoint, tmp_path / "evaluation_ms", split="test", device="cpu")
    assert result["count"] == 1
    assert all(key in result for key in (
        "rgb_psnr", "red_psnr", "nir_psnr", "ndvi_mae",
        "ndwi_mae", "ndbi_mae", "ndvi_mae_improvement_vs_bicubic",
    ))


def test_audit_crop_alignment_and_duplicates(tmp_path):
    manifest = fixture_manifest(tmp_path, size=64)
    report = prepare_manifest(manifest, tmp_path / "runtime.jsonl")
    assert report["counts"] == {"train": 1, "val": 1, "test": 1}
    for crop in (32, 64):
        config = config_for(manifest, tmp_path / "base")
        config["training"]["crop_size"] = crop
        data = dataset_for(config, "train")[0]
        assert data["lr"].shape[-1] == crop
        assert torch.equal(data["hr"], data["lr"].repeat_interleave(3, -1).repeat_interleave(3, -2))
        assert dataset_for(config, "val")[0]["lr"].shape[-1] == 64
    (tmp_path / "test.npz").write_bytes((tmp_path / "train.npz").read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_manifest(manifest, tmp_path / "duplicate.jsonl")


def test_training_resume_evaluation_and_adoption(tmp_path):
    manifest = fixture_manifest(tmp_path)
    base_config = config_for(manifest, tmp_path / "base")
    parent = train(base_config, "cpu")
    adopted_config = config_for(manifest, tmp_path / "adopted")
    adopted = adopt_base(adopted_config, parent)
    assert adopted.exists()
    assert adopt_base(adopted_config, parent) == adopted
    config = config_for(manifest, tmp_path / "residual", "sparse_transformer", parent)
    trained = train(config, "cpu")
    config["training"]["epochs"] = 2
    train(config, "cpu")
    last = torch.load(tmp_path / "residual/last.pt", weights_only=False)
    assert last["epoch"] == 2
    base_model, _ = load_model(parent)
    for key, tensor in base_model.base.state_dict().items():
        assert torch.equal(tensor, last["ema"][f"base.{key}"])
    metrics = evaluate(trained, tmp_path / "evaluation", split="test", device="cpu")
    assert metrics["count"] == 1 and np.isfinite(metrics["psnr"])
    assert evaluate(trained, tmp_path / "evaluation", split="test", device="cpu") == metrics
    rows = json.loads((tmp_path / "evaluation/per_image.json").read_text())
    assert len(paired_intervals(rows, replicates=10)) == 9
    timings = benchmark(trained, tmp_path / "timing.json", repeats=2, warmup=1, device="cpu")
    assert timings[1]["expert_tile_calls"] > 0
    assert timings[1]["mean_ms"] > 0
    changed = copy.deepcopy(config)
    changed["training"]["batch_size"] = 2
    with pytest.raises(ValueError, match="Epoch increases"):
        train(changed, "cpu")


def test_adversarial_training_step(tmp_path):
    manifest = fixture_manifest(tmp_path)
    parent = train(config_for(manifest, tmp_path / "base"), "cpu")
    config = config_for(manifest, tmp_path / "gan", "sparse_adversarial", parent)
    config["training"]["crop_size"] = 32
    assert train(config, "cpu").exists()


def test_notebook_experiment_workflow(tmp_path, monkeypatch):
    """Exercise the actual notebook cells on tiny CPU pairs, not just their syntax."""
    import ast
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import shutil
    import sys
    from geodiff_gan.experiments import trust_moe as workflow, trust_report as report
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "kaggle/GeoDiff_TrustMoE_Transformer_3x.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    manifest = fixture_manifest(tmp_path)
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "figures").mkdir()
    audit = prepare_manifest(manifest, suite / "runtime_manifest.jsonl")

    def run_command(command, cwd=None):
        config = json.loads(Path(command[-1]).read_text())
        config["training"]["num_workers"] = 0
        config["training"]["amp"] = False
        return workflow.train(config, "cpu")

    def fast_benchmark(*args, **kwargs):
        return report.benchmark(*args, **kwargs, repeats=2, warmup=1, device="cpu")

    # The production notebook remains full-data/full-epoch. Only the fixture is tiny.
    scope = dict(Path=Path, json=json, torch=torch, np=np, pd=pd, plt=plt, shutil=shutil,
        sys=sys, SUITE_ROOT=suite, REPOSITORY_DIR=root, MANIFEST=manifest, AUDIT=audit,
        REGION_FRACTION=0.5, TOP_K=2, NUM_EXPERTS=3, SEEDS=[42],
        EPOCHS=dict(base=1, residual=1), BATCH_SIZE=1, TRAIN_LR_CROP=32,
        BASE_MODEL=dict(width=8, base_embed_dim=8, base_heads=2, base_depth=1,
                        base_groups=1, router_depth=1),
        RUN={p: True for p in workflow.PROFILES}, IMPORT_BASE_CHECKPOINT=None,
        RUN_TEST_EVALUATION=True, DATASET_PROTOCOL="own_tiles", DISPLAY_MAX=0.3,
        display=lambda *args: None, run=run_command,
        make_config=workflow.make_config, write_json=workflow.write_json,
        digest_file=workflow.digest_file, adopt_base=workflow.adopt_base,
        evaluate=workflow.evaluate, dataset_for=workflow.dataset_for,
        METRICS=report.METRICS, summarize=report.summarize, benchmark=fast_benchmark,
        compare_controls=report.compare_controls, paired_intervals=report.paired_intervals,
        visualize=report.visualize, bundle_results=report.bundle_results)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    start = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("REGISTRY_PATH"):
            start = True
        if start and not source.startswith("evidence ="):
            source = source.replace("from IPython.display import Image as DisplayImage",
                                    "DisplayImage = lambda **kwargs: None")
            exec(compile(source, "<notebook-cell>", "exec"), scope)
    assert (suite / "test_plan.json").exists()
    assert (suite / "reports/test/controls_seed_42.csv").exists()
    assert len(scope["TEST_EVALS"]) == 9
    assert (suite / "figures/test_seed42_000000.png").exists()
    archive = report.bundle_results(suite, tmp_path / "results.zip", include_resume=True)
    import zipfile
    with zipfile.ZipFile(archive) as handle:
        assert "seed_42/sparse_transformer/last.pt" in handle.namelist()
        assert "bundle_inventory.json" in handle.namelist()
