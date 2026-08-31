"""Read-only saved-result explorer. Does not load a checkpoint or train a model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class SavedTileResults:
    def __init__(self, root):
        self.root = Path(root)
        state = json.loads((self.root / "suite_state.json").read_text(encoding="utf-8"))
        self.manifest = Path(state["manifest"])
        self.records = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.models = state["results"]

    def select(self, split, tile="All"):
        return [r for r in self.records if r["split"] == split and (tile == "All" or r["tile_id"] == tile)]

    @staticmethod
    def cache_name(record):
        path = Path(record["patch"])
        return "__".join((record["tile_id"], path.parent.name, path.stem)) + "_uncertainty.npz"

    def panels(self, index=0, split="test", names=None, show_base=False, tile="All",
               display_max=0.3, errors=False):
        records = self.select(split, tile)
        if not 0 <= int(index) < len(records):
            raise IndexError(f"Index {index} outside {split}/{tile}: 0..{len(records) - 1}")
        record = records[int(index)]
        with np.load(record["patch"]) as patch:
            lr, hr = patch["lr"].copy(), patch["hr"].copy()
            valid = patch["valid_mask_hr"][0].astype(bool)
        panels = [(lr, f"Landsat original 30 m | {lr.shape[-1]} x {lr.shape[-2]}")]
        details = {"record": record, "display": "Fixed common reflectance stretch; metrics use unchanged arrays", "models": {}}
        for name in (list(self.models) if names is None else names):
            root = Path(self.models[name]["root"]) / "evaluation" / split / "model"
            path = root / self.cache_name(record)
            if not path.exists():
                details["models"][name] = "No saved prediction for this index/split"
                continue
            with np.load(path) as data:
                image = data["mean"].copy()
                base = data["base"].copy()
                mse = float(((image - hr) ** 2)[:, valid].mean())
                base_mse = float(((base - hr) ** 2)[:, valid].mean())
                info = {"psnr": -10 * np.log10(max(mse, 1e-12)),
                        "l1": float(np.abs(image - hr)[:, valid].mean()),
                        "expert_weights": data["expert_weights"].tolist() if "expert_weights" in data else [],
                        "router_acceptance": float(data["router_acceptance"]) if "router_acceptance" in data else 1.0}
                if show_base:
                    info["base_psnr"] = -10 * np.log10(max(base_mse, 1e-12))
                    panels.append((base, f"{name}: base | PSNR={info['base_psnr']:.2f}"))
                panels.append((image, f"{name} | PSNR={info['psnr']:.2f}"))
                if errors:
                    error = np.abs(image - hr).mean(0)
                    error[~valid] = np.nan
                    panels.append((error, f"{name}: absolute error"))
                details["models"][name] = info
        panels.append((hr, f"Sentinel target 10 m | {hr.shape[-1]} x {hr.shape[-2]}"))
        return panels, details

    @staticmethod
    def display(image, maximum=0.3):
        return (np.clip(image.transpose(1, 2, 0) / max(float(maximum), 1e-6), 0, 1) ** (1 / 1.4) * 255).astype(np.uint8)


def build_app(root):
    import gradio as gr

    viewer = SavedTileResults(root)
    tiles = ["All", *sorted({record["tile_id"] for record in viewer.records})]

    def render(index, split, tile, names, base, maximum):
        records = viewer.select(split, tile)
        if not records:
            return [], {"message": "No patches in this filter"}, 0
        index = min(max(0, int(index)), len(records) - 1)
        panels, details = viewer.panels(index, split, names, base, tile, maximum)
        images = [(viewer.display(image, maximum), title) for image, title in panels]
        details["filtered_count"] = len(records)
        return images, details, index

    with gr.Blocks(title="GeoDiff tile experiments") as app:
        gr.Markdown("# Landsat / Sentinel experiment explorer\nSaved data only; no GPU or retraining. Shared display stretch is not used for metrics.")
        with gr.Row():
            split = gr.Dropdown(["train", "val", "test"], value="val", label="Split")
            tile = gr.Dropdown(tiles, value="All", label="Tile")
            index = gr.Number(value=0, precision=0, minimum=0, label="Index within filter")
        names = gr.CheckboxGroup(list(viewer.models), value=list(viewer.models), label="Models (empty = dataset only)")
        with gr.Row():
            base = gr.Checkbox(False, label="Show deterministic bases and base PSNR")
            maximum = gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="Display white point (all panels)")
            back = gr.Button("< Back")
            forward = gr.Button("Next >")
            refresh = gr.Button("Show")
        gallery = gr.Gallery(label="Original LR / selected models / target", columns=3, height="80vh", object_fit="contain")
        info = gr.JSON(label="Metrics, routing weights and pair metadata")
        inputs = [index, split, tile, names, base, maximum]
        outputs = [gallery, info, index]
        refresh.click(render, inputs, outputs)
        back.click(lambda i, *args: render(i - 1, *args), inputs, outputs)
        forward.click(lambda i, *args: render(i + 1, *args), inputs, outputs)
        split.change(lambda _, *args: render(0, *args), inputs, outputs)
        tile.change(lambda _, *args: render(0, *args), inputs, outputs)
        names.change(render, inputs, outputs)
        base.change(render, inputs, outputs)
        app.load(render, inputs, outputs)
    return app
