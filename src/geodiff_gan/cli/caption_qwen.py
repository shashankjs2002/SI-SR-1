from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..data.manifest import load_manifest

CAPTION_INSTRUCTION = """Describe this satellite image patch as compact JSON with keys:
land_cover, visible_objects, object_density, terrain, texture, confidence.
Only describe visible evidence. Do not infer coordinates, city names, country names,
ownership, people, or events. Keep each value short."""


def _load_image(path: str) -> Image.Image:
    with np.load(path) as data:
        array = data["hr"]
    if array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    array = (np.clip(array, 0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Caption prepared patches with Qwen3-VL")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    except ImportError as error:
        raise SystemExit(
            "Install geodiff-gan[caption] before running Qwen3-VL captioning"
        ) from error
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=quantization,
        dtype=torch.float16,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    records = load_manifest(args.manifest, split=args.split)
    if args.limit:
        records = records[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if output.exists():
        with output.open("r", encoding="utf-8") as handle:
            completed = {json.loads(line)["patch"] for line in handle if line.strip()}
    with output.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            if record.patch in completed:
                continue
            image = _load_image(record.patch)
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
            caption = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            handle.write(
                json.dumps(
                    {"patch": record.patch, "tile_id": record.tile_id, "caption": caption},
                    ensure_ascii=True,
                )
                + "\n"
            )
            handle.flush()
            if index % 25 == 0:
                print(f"captioned {index + 1}/{len(records)}", flush=True)


if __name__ == "__main__":
    main()
