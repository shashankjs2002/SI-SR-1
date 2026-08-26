from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from geodiff_gan.data.manifest import ManifestRecord, write_manifest


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def images_below(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def discover_root(input_root: Path) -> Path:
    candidates = []
    for candidate in (input_root, *input_root.rglob("*")):
        if not candidate.is_dir():
            continue
        if (candidate / "Train" / "HR").is_dir() and (candidate / "Train" / "LR").is_dir():
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find an ALSAT root containing Train/HR and Train/LR under {input_root}"
        )
    return min(candidates, key=lambda path: (len(path.parts), str(path)))


def casefold_index(directory: Path) -> dict[str, Path]:
    return {
        str(path.relative_to(directory)).replace("\\", "/").casefold(): path
        for path in images_below(directory)
    }


def paired_lr_path(
    hr_path: Path,
    hr_root: Path,
    lr_index: dict[str, Path],
    train: bool,
) -> Path | None:
    relative = hr_path.relative_to(hr_root)
    candidates = [relative]
    if train:
        candidates.extend(
            [
                relative.with_name(relative.name.replace("H", "L")),
                relative.with_name(relative.name.replace("h", "l")),
                relative.with_name(relative.name.replace("HR", "LR")),
                relative.with_name(relative.name.replace("hr", "lr")),
            ]
        )
    for candidate in candidates:
        key = str(candidate).replace("\\", "/").casefold()
        if key in lr_index:
            return lr_index[key]
    return None


def collect_pairs(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pairs: list[dict[str, object]] = []
    quarantine: list[dict[str, object]] = []

    train_hr = root / "Train" / "HR"
    train_lr = root / "Train" / "LR"
    train_lr_index = casefold_index(train_lr)
    for hr_path in images_below(train_hr):
        lr_path = paired_lr_path(hr_path, train_hr, train_lr_index, train=True)
        if lr_path is None:
            quarantine.append({"reason": "missing_train_lr", "hr": str(hr_path)})
            continue
        source_id = f"train/{hr_path.relative_to(train_hr).as_posix()}"
        pairs.append(
            {"source_id": source_id, "category": "Train", "hr": hr_path, "lr": lr_path}
        )

    test_root = root / "Test"
    for hr_root in sorted(path for path in test_root.rglob("HR") if path.is_dir()):
        lr_root = hr_root.parent / "LR"
        category = hr_root.parent.relative_to(test_root).as_posix() or "Test"
        if not lr_root.is_dir():
            quarantine.append({"reason": "missing_test_lr_directory", "hr_root": str(hr_root)})
            continue
        test_lr_index = casefold_index(lr_root)
        for hr_path in images_below(hr_root):
            lr_path = paired_lr_path(hr_path, hr_root, test_lr_index, train=False)
            if lr_path is None:
                quarantine.append({"reason": "missing_test_lr", "hr": str(hr_path)})
                continue
            source_id = f"test/{category}/{hr_path.relative_to(hr_root).as_posix()}"
            pairs.append(
                {"source_id": source_id, "category": category, "hr": hr_path, "lr": lr_path}
            )
    return pairs, quarantine


def stable_validation(source_id: str, percent: int) -> bool:
    bucket = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return bucket < percent


def read_rgb(path: Path, value_max: float) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return np.clip(array / value_max, 0.0, 1.0).transpose(2, 0, 1)


def crop_origins(height: int, width: int, size: int, limit: int) -> list[tuple[int, int]]:
    if height < size or width < size:
        return []
    rows = list(range(0, height - size + 1, size))
    cols = list(range(0, width - size + 1, size))
    if rows[-1] != height - size:
        rows.append(height - size)
    if cols[-1] != width - size:
        cols.append(width - size)
    origins = [(row, col) for row in rows for col in cols]
    if limit > 0 and len(origins) > limit:
        indices = np.linspace(0, len(origins) - 1, limit).round().astype(int)
        origins = [origins[index] for index in sorted(set(indices.tolist()))]
    return origins


def apply_source_limit(
    pairs: list[dict[str, object]], split: str, limit: int | None
) -> list[dict[str, object]]:
    selected = [pair for pair in pairs if pair["split"] == split]
    return selected if limit is None else selected[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare native 4x ALSAT pairs for GeoDiff-GAN")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--quarantine", required=True)
    parser.add_argument("--lr-crop-size", type=int, default=32)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--train-patches-per-source", type=int, default=16)
    parser.add_argument("--eval-patches-per-source", type=int, default=4)
    parser.add_argument("--max-train-sources", type=int)
    parser.add_argument("--max-val-sources", type=int)
    parser.add_argument("--max-test-sources", type=int)
    parser.add_argument("--value-max", type=float, default=255.0)
    args = parser.parse_args()

    if not 1 <= args.validation_percent < 100:
        raise ValueError("--validation-percent must be in [1, 99]")
    if args.scale != 4:
        raise ValueError("The native ALSAT protocol used here is exactly 4x")
    if args.value_max <= 0:
        raise ValueError("--value-max must be positive")

    root = discover_root(Path(args.input).resolve())
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    quarantine_path = Path(args.quarantine).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    pairs, quarantine = collect_pairs(root)
    for pair in pairs:
        pair["split"] = (
            "test"
            if str(pair["source_id"]).startswith("test/")
            else "val"
            if stable_validation(str(pair["source_id"]), args.validation_percent)
            else "train"
        )

    selected: list[dict[str, object]] = []
    selected.extend(apply_source_limit(pairs, "train", args.max_train_sources))
    selected.extend(apply_source_limit(pairs, "val", args.max_val_sources))
    selected.extend(apply_source_limit(pairs, "test", args.max_test_sources))

    records: list[ManifestRecord] = []
    split_sources: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for pair in tqdm(selected, desc="ALSAT source pairs", unit="source"):
        source_id = str(pair["source_id"])
        split = str(pair["split"])
        try:
            lr = read_rgb(Path(pair["lr"]), args.value_max)
            hr = read_rgb(Path(pair["hr"]), args.value_max)
        except Exception as error:
            quarantine.append({"reason": "read_error", "source_id": source_id, "error": repr(error)})
            continue
        if hr.shape[-2:] != (lr.shape[-2] * args.scale, lr.shape[-1] * args.scale):
            quarantine.append(
                {
                    "reason": "geometry_mismatch",
                    "source_id": source_id,
                    "lr_shape": list(lr.shape),
                    "hr_shape": list(hr.shape),
                }
            )
            continue
        patch_limit = (
            args.train_patches_per_source if split == "train" else args.eval_patches_per_source
        )
        origins = crop_origins(lr.shape[-2], lr.shape[-1], args.lr_crop_size, patch_limit)
        if not origins:
            quarantine.append({"reason": "source_too_small", "source_id": source_id})
            continue
        source_hash = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]
        for row, col in origins:
            hr_row, hr_col = row * args.scale, col * args.scale
            lr_crop = lr[:, row : row + args.lr_crop_size, col : col + args.lr_crop_size]
            hr_size = args.lr_crop_size * args.scale
            hr_crop = hr[:, hr_row : hr_row + hr_size, hr_col : hr_col + hr_size]
            patch_path = output / split / f"{source_hash}_r{row:04d}_c{col:04d}.npz"
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            if not patch_path.exists():
                np.savez_compressed(
                    patch_path,
                    lr=lr_crop.astype(np.float16),
                    clean_lr=lr_crop.astype(np.float16),
                    hr=hr_crop.astype(np.float16),
                    valid_mask_lr=np.ones((1, *lr_crop.shape[-2:]), dtype=np.uint8),
                    valid_mask_hr=np.ones((1, *hr_crop.shape[-2:]), dtype=np.uint8),
                    degradation=np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32),
                    source_lr=str(pair["lr"]),
                    source_hr=str(pair["hr"]),
                    category=str(pair["category"]),
                )
            records.append(
                ManifestRecord(
                    patch=str(patch_path),
                    tile_id=source_id,
                    split=split,
                    row=hr_row,
                    col=hr_col,
                    valid_fraction=1.0,
                    source="alsat_2b_paired_benchmark",
                    license_id="verify-alsat-2b-dataset-license",
                    source_product=source_id,
                    scale=args.scale,
                )
            )
            split_sources[split].add(source_id)

    if not records:
        raise RuntimeError("No valid ALSAT patches were produced")
    if not split_sources["train"] or not split_sources["val"] or not split_sources["test"]:
        raise RuntimeError(
            "ALSAT preparation needs non-empty train, val and test source sets; "
            f"found { {key: len(value) for key, value in split_sources.items()} }"
        )
    write_manifest(manifest_path, records)
    with quarantine_path.open("w", encoding="utf-8") as handle:
        for row in quarantine:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "root": str(root),
        "sources": {key: len(value) for key, value in split_sources.items()},
        "patches": {
            key: sum(record.split == key for record in records)
            for key in ("train", "val", "test")
        },
        "quarantined": len(quarantine),
        "manifest": str(manifest_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
