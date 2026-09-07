"""Measured quality, routing, speed and export artifacts for TrustMoE."""
from __future__ import annotations

import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .trust_moe import dataset_for, load_model, write_json, digest_file


METRICS = ("psnr", "ssim", "edge_f1", "l1", "ergas", "sam_degrees", "uiqi", "scc", "redegradation_l1")
LOWER = {"l1", "ergas", "sam_degrees", "redegradation_l1"}


def paired_intervals(rows, replicates=2000, seed=42):
    """Pair bootstrap is descriptive; tile-cluster intervals supplied when possible."""
    rng = np.random.default_rng(seed)
    n = len(rows)
    indices = rng.integers(0, n, (replicates, n))
    groups = sorted({r["tile_id"] for r in rows})
    result = []
    for metric in METRICS:
        delta = np.array([r[metric] - r[f"base_{metric}"] for r in rows])
        if metric in LOWER:
            delta *= -1
        interval = np.quantile(delta[indices].mean(1), [0.025, 0.975])
        cluster_interval = [None, None]
        if len(groups) >= 2:
            sums = np.array([sum(d for r, d in zip(rows, delta) if r["tile_id"] == g) for g in groups])
            counts = np.array([sum(r["tile_id"] == g for r in rows) for g in groups])
            chosen = rng.integers(0, len(groups), (replicates, len(groups)))
            cluster_interval = np.quantile(sums[chosen].sum(1) / counts[chosen].sum(1), [0.025, 0.975]).tolist()
        result.append({"metric": metric, "mean_improvement": float(delta.mean()),
                       "ci95_low": float(interval[0]), "ci95_high": float(interval[1]),
                       "tile_cluster_ci95_low": cluster_interval[0], "tile_cluster_ci95_high": cluster_interval[1],
                       "independent_tile_groups": len(groups), "images_better": int((delta > 0).sum()),
                       "fraction_better": float((delta > 0).mean())})
    return result


def summarize(evaluations, output_root):
    import pandas as pd
    import matplotlib.pyplot as plt
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    overall, per_class, intervals, routes = [], [], [], []
    for name, path in evaluations.items():
        path = Path(path)
        summary = json.loads((path / "metrics.json").read_text())
        rows = json.loads((path / "per_image.json").read_text())
        overall.append({**summary, "experiment": name})
        intervals.extend({"experiment": name, **row} for row in paired_intervals(rows))
        for category in sorted({r["scene_class"] for r in rows}):
            selected = [r for r in rows if r["scene_class"] == category]
            per_class.append({"experiment": name, "scene_class": category, "count": len(selected),
                              **{k: float(np.mean([r[k] for r in selected])) for k in METRICS}})
            load = np.sum([r["expert_load"] for r in selected], axis=0)
            for expert, count in enumerate(load):
                routes.append({"experiment": name, "scene_class": category, "expert": expert,
                               "tile_assignments": int(count), "fraction": float(count / max(load.sum(), 1))})
    if overall:
        reference = overall[0]
        overall.insert(0, {"experiment": "bicubic", "count": reference["count"],
                           **{key: reference[f"bicubic_{key}"] for key in METRICS}})
    tables = {"overall": pd.DataFrame(overall), "per_class": pd.DataFrame(per_class),
              "paired_intervals": pd.DataFrame(intervals), "expert_usage": pd.DataFrame(routes)}
    for name, table in tables.items():
        table.to_csv(root / f"{name}.csv", index=False)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, metric in zip(axes.flat, ("psnr", "ssim", "edge_f1", "scc")):
        table = tables["overall"]
        axis.bar(np.arange(len(table)), table[metric], color="#296b8b")
        axis.set_xticks(np.arange(len(table)), table.experiment, rotation=35, ha="right", fontsize=8)
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(root / "quality_comparison.png", dpi=180)
    plt.close(figure)
    return tables


def compare_controls(evaluations, selected, output_path):
    """Paired comparisons against controls, refusing mismatched input ordering."""
    selected_root = Path(evaluations[selected])
    selected_request = json.loads((selected_root / "request.json").read_text())
    selected_rows = json.loads((selected_root / "per_image.json").read_text())
    tables = []
    for name, path in evaluations.items():
        if name == selected:
            continue
        request = json.loads((Path(path) / "request.json").read_text())
        rows = json.loads((Path(path) / "per_image.json").read_text())
        if any(request[key] != selected_request[key] for key in ("dataset_id", "split")):
            raise ValueError(f"{name}: incomparable dataset or split")
        if [r["patch"] for r in rows] != [r["patch"] for r in selected_rows]:
            raise ValueError(f"{name}: input pairs differ")
        paired = [{**chosen, **{f"base_{m}": control[m] for m in METRICS}}
                  for chosen, control in zip(selected_rows, rows)]
        tables.extend({"selected": selected, "control": name, **r} for r in paired_intervals(paired))
    import pandas as pd
    table = pd.DataFrame(tables)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


@torch.inference_mode()
def benchmark(checkpoint, output_path, *, manifest=None, repeats=30, warmup=5,
              coverage=None, top_k=None, device=None, frames=5):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_model(checkpoint, device)
    if manifest:
        config["manifest"] = str(manifest)
    dataset = dataset_for(config, "val")
    if not len(dataset) or repeats < 1 or warmup < 0 or frames < 1:
        raise ValueError("Need nonempty validation data and positive timing repetitions/frames")
    indices = np.linspace(0, len(dataset) - 1, min(len(dataset), frames)).astype(int).tolist()
    inputs = [dataset[i]["lr"][None].to(device) for i in indices]
    lr = inputs[0]
    amp = bool(config["training"]["amp"] and device.type == "cuda")
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    rows = []
    for name, base_only in (("base", True), ("full", False)):
        base_only = base_only or config["profile"] == "base"
        def run(index=0):
            with torch.autocast(device.type, enabled=amp):
                return model(inputs[index % len(inputs)], base_only=base_only, coverage=coverage, top_k=top_k)
        for i in range(warmup):
            run(i)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timings, executed = [], []
        for i in range(repeats):
            sync()
            start = time.perf_counter()
            result = run(i)
            sync()
            timings.append(1000 * (time.perf_counter() - start))
            executed.append(int(result.dispatched_tiles))
            del result
        result = run()
        modules = {k: sum(p.numel() for p in m.parameters()) for k, m in model.named_children()}
        expert_sizes = [sum(p.numel() for p in m.parameters()) for m in model.experts]
        selected = (result.assignments.sum((0, 1)) > 0).cpu().tolist()
        active_params = modules["base"] if base_only else sum(modules.values()) - modules["experts"] + sum(v for v, use in zip(expert_sizes, selected) if use)
        rows.append({"operation": name, "mean_ms": float(np.mean(timings)),
                     "median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
                     "images_per_second": 1000 / float(np.mean(timings)),
                     "parameters_total": sum(modules.values()), "parameters_used_on_this_frame": active_params,
                     "components": modules, "expert_tile_calls": int(result.dispatched_tiles),
                     "active_fraction": float(result.active.mean()),
                     "mean_timed_expert_tile_calls": float(np.mean(executed)),
                     "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                     "repeats": repeats, "warmup": warmup, "amp": amp, "batch_size": 1,
                     "input_shape": list(lr.shape), "device": str(device),
                     "validation_indices": indices, "input_shapes": [list(x.shape) for x in inputs],
                     "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                     "torch": torch.__version__, "coverage": coverage, "top_k": top_k,
                     "scope": "resident model forward including routing/dispatch; excludes file I/O and host-to-device copy",
                     "memory_scope": "full model and sampled LR inputs resident for both operations; allocator peak reset after warmup",
                     "flops": None, "flops_note": "Not estimated from top-k. Use dispatched work and measured latency; attention/gather/scatter also cost time."})
        del result
    write_json(output_path, rows)
    return rows


def visualize(checkpoints, manifest, index=0, split="test", show_base=True,
              output_path=None, display_max=0.3, gamma=1.4, coverage=None, top_k=None):
    import matplotlib.pyplot as plt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first = next(iter(checkpoints.values()))
    model, config = load_model(first, device)
    config["manifest"] = str(manifest)
    dataset = dataset_for(config, split)
    if not 0 <= index < len(dataset):
        raise IndexError(f"{split} index must be 0..{len(dataset)-1}")
    sample = dataset[index]
    lr, target = sample["lr"][None].to(device), sample["hr"]
    panels = [(sample["lr"].numpy(), "Landsat LR"), (target.numpy(), "Sentinel HR")]
    routing_figures = []
    for i, (name, path) in enumerate(checkpoints.items()):
        if i:
            model, config = load_model(path, device)
        with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda" and config["training"]["amp"]):
            result = model(lr, coverage=coverage, top_k=top_k, base_only=config["profile"] == "base")
        if i == 0 and show_base:
            panels.append((result.base[0].float().cpu().numpy(), "Shared base"))
        panels.append((result.image[0].float().cpu().numpy(), name))
        routing_figures.append((name, result.active[0, 0].cpu().numpy(), result.trust[0].float().mean(0).cpu().numpy()))
        del model, result
    figure, axes = plt.subplots(int(np.ceil(len(panels) / 3)), 3,
                                figsize=(15, 5 * int(np.ceil(len(panels) / 3))), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (array, title) in zip(axes.flat, panels):
        rgb = np.clip(array[:3].transpose(1, 2, 0) / display_max, 0, 1) ** (1 / gamma)
        axis.imshow(rgb, interpolation="nearest")
        axis.set_title(f"{title}\n{array.shape[-1]} x {array.shape[-2]}", fontsize=10)
    figure.suptitle(f"{split}/{index} | common display stretch only; metrics use unchanged tensors")
    figure.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=170)
    plt.show()
    maps, axes = plt.subplots(len(routing_figures), 2, figsize=(10, 4 * len(routing_figures)), squeeze=False)
    for row, (name, active, trust) in enumerate(routing_figures):
        for axis, image, title in zip(axes[row], (active, trust), ("Executed regions", "Mean RGB trust")):
            axis.imshow(image, vmin=0, vmax=1, cmap="viridis")
            axis.set_title(f"{name}: {title}")
            axis.axis("off")
    maps.tight_layout()
    if output_path:
        maps.savefig(Path(output_path).with_name(Path(output_path).stem + "_routing.png"), dpi=170)
    plt.show()


def visualize_saved(evaluations, manifest, index=0, split="test", show_base=True,
                    output_path=None, display_max=0.3, gamma=1.4):
    """Display the exact evaluated tensors, preserving each model's own compute budget."""
    import matplotlib.pyplot as plt
    from ..data.dataset import SentinelPatchDataset
    dataset = SentinelPatchDataset(manifest, split, scale=3, input_mode="paired",
                                    augment=False, random_degradation=False)
    if not 0 <= index < len(dataset):
        raise IndexError(f"{split} index must be 0..{len(dataset)-1}")
    sample = dataset[index]
    panels = [(sample["lr"].numpy(), "Original Landsat LR"), (sample["hr"].numpy(), "Sentinel HR")]
    maps = []
    for i, (name, path) in enumerate(evaluations.items()):
        path = Path(path)
        receipt = json.loads((path / "request.json").read_text())
        if receipt["split"] != split:
            raise ValueError(f"{name}: saved outputs are for another split")
        row = json.loads((path / "records" / f"{index:06d}.json").read_text())
        if Path(row["patch"]).name != Path(sample["patch"]).name:
            raise ValueError(f"{name}: saved prediction does not match this dataset index")
        with np.load(path / "images" / f"{index:06d}.npz", allow_pickle=False) as values:
            if i == 0 and show_base:
                panels.append((values["base"], f"Shared base | PSNR {row['base_psnr']:.3f}"))
            panels.append((values["prediction"], f"{name} | PSNR {row['psnr']:.3f}"))
            assignments = values["assignments"]
            dominant = np.argmax(values["probabilities"] * assignments, axis=-1).astype(float)
            dominant[assignments.sum(-1) == 0] = np.nan
            maps.append((name, values["active"][0], values["trust"].mean(0), dominant.reshape(tuple(values["token_grid"]))))
    figure, axes = plt.subplots(int(np.ceil(len(panels) / 3)), 3,
                                figsize=(15, 4.7 * int(np.ceil(len(panels) / 3))), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (array, title) in zip(axes.flat, panels):
        axis.imshow(np.clip(array.transpose(1, 2, 0) / display_max, 0, 1) ** (1 / gamma), interpolation="nearest")
        axis.set_title(f"{title}\n{array.shape[-1]} x {array.shape[-2]}", fontsize=10)
    figure.suptitle(f"{split}/{index}: exact saved predictions; shared display stretch only")
    figure.tight_layout()
    route_figure, axes = plt.subplots(len(maps), 3, figsize=(14, 4 * len(maps)), squeeze=False)
    for row, (name, active, trust, dominant) in enumerate(maps):
        for column, (axis, array, label) in enumerate(zip(axes[row], (active, trust, dominant), ("Executed regions", "Mean RGB trust", "Dominant expert (blank = bypass)"))):
            axis.imshow(array, cmap="viridis" if column < 2 else "tab10", interpolation="nearest",
                        vmin=0, vmax=1 if column < 2 else max(1, assignments.shape[-1] - 1))
            axis.set_title(f"{name}\n{label}", fontsize=10)
            axis.axis("off")
    route_figure.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=170)
        route_figure.savefig(output_path.with_name(output_path.stem + "_routes.png"), dpi=170)
    plt.show()
    return figure, route_figure


def bundle_results(suite, archive, source_root=None, include_resume=True):
    """Bundle reproducible evidence, trained best models and receipts, excluding raw tiles/secrets."""
    suite, archive = Path(suite), Path(archive)
    included = []
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(suite.rglob("*")):
            if not path.is_file() or path == archive or path.suffix in (".zip", ".tmp"):
                continue
            relative = path.relative_to(suite)
            if any(part in ("raw", "patches", ".venvs", ".git", "__pycache__") for part in relative.parts):
                continue
            if path.suffix == ".pt" and path.name not in (("best.pt", "last.pt") if include_resume else ("best.pt",)):
                continue
            if path.name in ("kaggle.json", ".env"):
                continue
            handle.write(path, relative.as_posix())
            included.append({"file": relative.as_posix(), "sha256": digest_file(path), "bytes": path.stat().st_size})
        if source_root:
            source_root = Path(source_root)
            for folder in ("src", "configs", "scripts"):
                for path in sorted((source_root / folder).rglob("*")):
                    if path.is_file() and path.suffix in (".py", ".yaml"):
                        name = "source/" + path.relative_to(source_root).as_posix()
                        handle.write(path, name)
                        included.append({"file": name, "sha256": digest_file(path), "bytes": path.stat().st_size})
        handle.writestr("bundle_inventory.json", json.dumps(included, indent=2))
    return archive
