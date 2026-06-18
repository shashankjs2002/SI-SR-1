from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..data.manifest import load_manifest

CAPTION_INSTRUCTION = """You are captioning a Sentinel-2 RGB satellite image patch for
prompt-conditioned super-resolution research.

Return strict JSON only. Do not wrap it in markdown.

Required JSON schema:
{
  "brief": "one short evidence-only phrase, maximum 18 words",
  "descriptive": "one or two neutral sentences describing visible land cover, objects, density, terrain, and texture",
  "analytical": {
    "land_cover": ["visible land-cover classes only"],
    "visible_objects": ["visible object or structure categories only"],
    "object_density": "sparse | moderate | dense | mixed | unclear",
    "terrain": "flat | hilly | mountainous | coastal | riverine | arid | mixed | unclear",
    "texture": "smooth | fine-grained | coarse | grid-like | linear | mixed | unclear",
    "spatial_layout": "short description of dominant spatial arrangement",
    "uncertainty": "low | medium | high"
  }
}

Rules:
- Only describe visible evidence in the image.
- Do not infer coordinates, city names, country names, ownership, people, events, or time.
- Do not mention exact sensor/product metadata.
- If uncertain, use "unclear" or "possibly" rather than inventing details.
- Prefer remote-sensing terms such as agricultural fields, built-up area, roads, river channel,
  scrubland, bare soil, water body, forest, settlement, ridge, quarry, cloud, or haze when visible."""


def _resolve_model_class(transformers_module: Any) -> type:
    """Resolve Qwen3-VL across supported Transformers API generations."""
    for name in (
        "Qwen3VLForConditionalGeneration",
        "AutoModelForMultimodalLM",
        "AutoModelForVision2Seq",
    ):
        model_class = getattr(transformers_module, name, None)
        if model_class is not None:
            return model_class
    raise RuntimeError(
        "The installed Transformers build does not expose a Qwen3-VL-compatible "
        "conditional-generation class. Install the caption dependencies from "
        "pyproject.toml or requirements-kaggle.txt."
    )


def _resolve_patch_path(
    patch: str,
    root: str | Path | None = None,
    manifest: str | Path | None = None,
) -> Path:
    patch_path = Path(patch)
    candidates = []
    if patch_path.is_absolute():
        candidates.append(patch_path)
    if root is not None:
        candidates.append(Path(root) / patch_path)
    if manifest is not None:
        candidates.append(Path(manifest).resolve().parent / patch_path)
    candidates.append(patch_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_image(
    path: str,
    root: str | Path | None = None,
    manifest: str | Path | None = None,
    display_gain: float = 2.5,
) -> Image.Image:
    patch_path = _resolve_patch_path(path, root=root, manifest=manifest)
    with np.load(patch_path) as data:
        array = data["hr"]
    if array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    array = (np.clip(array * display_gain, 0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _analytical_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    parts: list[str] = []
    land_cover = value.get("land_cover")
    if isinstance(land_cover, list) and land_cover:
        parts.append("land cover: " + ", ".join(map(str, land_cover[:6])))
    visible_objects = value.get("visible_objects")
    if isinstance(visible_objects, list) and visible_objects:
        parts.append("objects: " + ", ".join(map(str, visible_objects[:8])))
    for key in ("object_density", "terrain", "texture", "spatial_layout", "uncertainty"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            parts.append(f"{key.replace('_', ' ')}: {item.strip()}")
    return "; ".join(parts)


def _caption_text(payload: dict[str, Any], preferred: str) -> str:
    value = payload.get(preferred)
    if preferred == "analytical":
        text = _analytical_to_text(value)
    else:
        text = value if isinstance(value, str) else ""
    if text:
        return text.strip()
    for fallback in ("descriptive", "brief", "analytical"):
        if fallback == preferred:
            continue
        value = payload.get(fallback)
        text = _analytical_to_text(value) if fallback == "analytical" else value
        if isinstance(text, str) and text.strip():
            return text.strip()
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Caption prepared patches with Qwen3-VL")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--root",
        help="Dataset root used to resolve relative patch paths from a Colab/Kaggle dataset.",
    )
    parser.add_argument(
        "--preferred-caption",
        choices=("brief", "descriptive", "analytical"),
        default="descriptive",
        help="Caption variant copied into the backward-compatible top-level caption field.",
    )
    parser.add_argument(
        "--display-gain",
        type=float,
        default=2.5,
        help="Brightness gain applied only to the image shown to the VLM.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import torch
        import transformers
        from transformers import AutoProcessor, BitsAndBytesConfig
    except ImportError as error:
        raise SystemExit(
            "Install geodiff-gan[caption] before running Qwen3-VL captioning"
        ) from error
    model_class = _resolve_model_class(transformers)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = model_class.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=quantization,
        dtype=torch.float16,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    records = load_manifest(args.manifest, split=None if args.split == "all" else args.split)
    if args.limit:
        records = records[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if output.exists():
        with output.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    completed.add(json.loads(line)["patch"])
                except json.JSONDecodeError:
                    continue
    with output.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            if record.patch in completed:
                continue
            image = _load_image(
                record.patch,
                root=args.root,
                manifest=args.manifest,
                display_gain=args.display_gain,
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": CAPTION_INSTRUCTION},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            trimmed = generated[:, inputs["input_ids"].shape[-1] :]
            response = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            payload = _parse_json_object(response) or {"descriptive": response}
            caption = _caption_text(payload, args.preferred_caption)
            handle.write(
                json.dumps(
                    {
                        "patch": record.patch,
                        "tile_id": record.tile_id,
                        "caption": caption,
                        "captions": payload,
                        "preferred_caption": args.preferred_caption,
                        "source_model": args.model,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            handle.flush()
            if index % 25 == 0:
                print(f"captioned {index + 1}/{len(records)}", flush=True)


if __name__ == "__main__":
    main()
