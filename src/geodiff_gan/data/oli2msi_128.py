"""Prepare the official OLI2MSI benchmark for 128 -> 384 TrustMoE experiments."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .manifest import ManifestRecord, load_manifest, write_manifest


PROTOCOL = "oli2msi_official_rgb_clip03_uint8_center128_v1"
REQUIRED = ("train_lr", "train_hr", "test_lr", "test_hr")


def _normalized(path):
    return "".join(character for character in Path(path).name.casefold() if character.isalnum())


def _tiffs(path):
    return sorted(item for item in Path(path).iterdir()
                  if item.is_file() and item.suffix.casefold() in (".tif", ".tiff"))


def find_oli2msi_layout(start):
    """Accept the dataset root, a split/LR/HR directory, or one source TIFF."""
    start = Path(start)
    if start.is_file():
        start = start.parent
    if not start.exists():
        raise FileNotFoundError(start)
    candidates = []
    current = start
    for _ in range(6):
        candidates.append(current)
        if current.parent == current:
            break
        current = current.parent
    ancestors = list(candidates)
    for parent in ancestors:
        try:
            candidates.extend(child for child in parent.iterdir() if child.is_dir())
        except PermissionError:
            pass
    candidates.extend(path for path in start.rglob("*") if path.is_dir())
    by_parent = {}
    names = {"trainlr": "train_lr", "trainhr": "train_hr",
             "testlr": "test_lr", "testhr": "test_hr"}
    for directory in candidates:
        key = names.get(_normalized(directory))
        if key:
            by_parent.setdefault(directory.parent.resolve(), {})[key] = directory.resolve()
    layouts = [(parent, dirs) for parent, dirs in by_parent.items()
               if set(dirs) == set(REQUIRED) and all(_tiffs(dirs[key]) for key in REQUIRED)]
    unique = {str(parent): (parent, dirs) for parent, dirs in layouts}
    if len(unique) != 1:
        summary = {str(parent): {key: len(_tiffs(value)) for key, value in dirs.items()}
                   for parent, dirs in unique.values()}
        raise RuntimeError(f"Expected exactly one complete OLI2MSI layout, found {len(unique)}: {summary}")
    return next(iter(unique.values()))


def _pairs(lr_directory, hr_directory):
    lr = {path.stem.casefold(): path for path in _tiffs(lr_directory)}
    hr = {path.stem.casefold(): path for path in _tiffs(hr_directory)}
    if len(lr) != len(_tiffs(lr_directory)) or len(hr) != len(_tiffs(hr_directory)):
        raise ValueError("Duplicate case-insensitive TIFF stems")
    return ([(key, lr[key], hr[key]) for key in sorted(lr.keys() & hr.keys())],
            sorted(lr.keys() - hr.keys()), sorted(hr.keys() - lr.keys()))


def _split(key, seed, validation_percent):
    bucket = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) % 100
    return "val" if bucket < validation_percent else "train"


def prepare_oli2msi_128(source, output, *, seed=42, validation_percent=5,
                        reflectance_max=0.3, quantize=True, fast_limit=None,
                        require_official_counts=True):
    """Keep train sources full for random crops; center-crop val/test to 128 -> 384."""
    import rasterio

    if not 0 < validation_percent < 100 or reflectance_max <= 0:
        raise ValueError("Invalid validation percentage or reflectance maximum")
    root, directories = find_oli2msi_layout(source)
    train, train_without_hr, train_without_lr = _pairs(directories["train_lr"], directories["train_hr"])
    test, test_without_hr, test_without_lr = _pairs(directories["test_lr"], directories["test_hr"])
    missing = dict(train_without_hr=train_without_hr, train_without_lr=train_without_lr,
                   test_without_hr=test_without_hr, test_without_lr=test_without_lr)
    if require_official_counts and (len(train) != 5225 or len(test) != 100):
        raise RuntimeError(f"Official run requires 5,225 train and 100 test pairs; found {len(train)}, {len(test)}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(protocol=PROTOCOL, source_root=str(root), seed=seed,
                    validation_percent=validation_percent, reflectance_max=reflectance_max,
                    quantize=quantize, train_source_shape=[160, 480],
                    model_shape=[128, 384], missing=missing)
    lock = output / "protocol.json"
    if lock.exists() and json.loads(lock.read_text()) != protocol:
        raise ValueError("OLI2MSI protocol changed; use a new output directory")
    temporary = lock.with_suffix(".tmp")
    temporary.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    temporary.replace(lock)
    if any(missing.values()):
        (output / "unpaired.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")

    selected = {"train": [], "val": [], "test": test}
    for pair in train:
        selected[_split(pair[0], seed, validation_percent)].append(pair)
    if fast_limit is not None:
        selected = {key: values[:int(fast_limit)] for key, values in selected.items()}
    if any(not selected[key] for key in selected):
        raise ValueError({key: len(value) for key, value in selected.items()})

    audit_paths = [path for values in selected.values() for _, lr, hr in values[:3] for path in (lr, hr)]
    p99 = []
    for path in audit_paths:
        with rasterio.open(path) as source_image:
            p99.append(float(np.nanpercentile(source_image.read([1, 2, 3], out_dtype="float32"), 99)))
    divisor = 1.0 if max(p99) <= 1.5 else 10000.0 if max(p99) <= 12000 else None
    if divisor is None:
        raise ValueError(f"Unsupported source range: p99={max(p99)}")

    def read(path, expected):
        with rasterio.open(path) as source_image:
            if source_image.count < 3 or source_image.shape != (expected, expected):
                raise ValueError(f"Unexpected {path}: bands={source_image.count}, shape={source_image.shape}")
            array = source_image.read([1, 2, 3], out_dtype="float32") / divisor
            mask = (source_image.read_masks([1, 2, 3]) > 0).all(0) & np.isfinite(array).all(0)
        array = np.clip(np.nan_to_num(array), 0, reflectance_max) / reflectance_max
        if quantize:
            array = np.round(array * 255) / 255
        return array, mask

    manifest = output / "manifest.jsonl"
    existing = {record.source_product: record for record in load_manifest(manifest) if Path(record.patch).is_file()} \
        if manifest.exists() else {}
    records, status = {}, Counter()
    for split, pairs in selected.items():
        for key, lr_path, hr_path in pairs:
            source_key = f"{split}:{key}"
            if source_key in existing:
                records[source_key] = existing[source_key]
                status["skipped"] += 1
                continue
            try:
                lr, ml = read(lr_path, 160)
                hr, mh = read(hr_path, 480)
                row = col = 0
                if split != "train":
                    row = col = 16
                    lr = lr[:, 16:144, 16:144]
                    ml = ml[16:144, 16:144]
                    hr = hr[:, 48:432, 48:432]
                    mh = mh[48:432, 48:432]
                valid = min(float(ml.mean()), float(mh.mean()))
                if valid < .99:
                    raise ValueError(f"valid_fraction={valid:.6f}")
                path = output / "patches" / split / f"{key}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, lr=lr.astype(np.float16), hr=hr.astype(np.float16),
                                    valid_mask_lr=ml[None].astype(np.uint8),
                                    valid_mask_hr=mh[None].astype(np.uint8))
                tmp.replace(path)
                records[source_key] = ManifestRecord(patch=str(path.resolve()), tile_id=key,
                    split=split, row=row, col=col, valid_fraction=valid, source=PROTOCOL,
                    license_id="official-OLI2MSI-distribution", source_product=source_key,
                    landsat_product=lr_path.name, sentinel_product=hr_path.name, scale=3)
                status["written"] += 1
            except Exception as error:
                status["quarantined"] += 1
                quarantine = output / "quarantine.jsonl"
                with quarantine.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(split=split, key=key, lr=str(lr_path), hr=str(hr_path),
                                                 error=f"{type(error).__name__}: {error}")) + "\n")
            if sum(status.values()) % 100 == 0:
                write_manifest(manifest, sorted(records.values(), key=lambda r: (r.split, r.tile_id)))
    write_manifest(manifest, sorted(records.values(), key=lambda r: (r.split, r.tile_id)))
    expected = {f"{split}:{key}" for split, values in selected.items() for key, _, _ in values}
    if set(records) != expected:
        raise RuntimeError("Quarantined/missing pairs prevent an exact benchmark; inspect quarantine.jsonl")
    counts = Counter(record.split for record in records.values())
    card = dict(protocol=protocol, counts=dict(counts), divisor=divisor,
                training="random aligned 128->384 crop from each 160->480 source per epoch",
                evaluation="fixed center 128->384 crop", status=dict(status))
    (output / "dataset_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return manifest, card
