"""Category metadata and portable packaging for paired satellite patches."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import shutil

from .manifest import ManifestRecord, load_manifest


SCENE_CLASS_ALIASES = {
    "urban": ("urban", "built_up", "built-up", "city"),
    "agriculture": ("agriculture", "agricultural", "cropland", "farm"),
    "forest": ("forest", "woodland"),
    "water_wetland": ("water", "wetland", "river", "lake"),
    "barren": ("barren", "desert", "bare_soil"),
    "mountain": ("mountain", "hilly", "highland"),
    "mixed_rural": ("mixed_rural", "mixed-rural", "rural"),
}


def normalize_scene_class(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
    if not normalized:
        raise ValueError("Scene class cannot be empty")
    for canonical, aliases in SCENE_CLASS_ALIASES.items():
        if normalized == canonical or normalized in {
            re.sub(r"[^a-z0-9]+", "_", alias.casefold()).strip("_")
            for alias in aliases
        }:
            return canonical
    return normalized


def infer_scene_class(path: str | Path) -> str | None:
    """Infer a conservative class only from directory names, never image pixels."""
    parts = [normalize_scene_class(part) for part in Path(path).parts if part]
    matches = []
    for canonical, aliases in SCENE_CLASS_ALIASES.items():
        accepted = {canonical, *(normalize_scene_class(alias) for alias in aliases)}
        if any(part in accepted for part in parts):
            matches.append(canonical)
    if len(matches) > 1:
        raise ValueError(f"Ambiguous scene class in {path}: {sorted(matches)}")
    return matches[0] if matches else None


def annotate_scene_classes(
    records: list[ManifestRecord],
    *,
    category_by_tile: dict[str, str] | None = None,
    category_by_product: dict[str, str] | None = None,
    require_all: bool = True,
) -> list[ManifestRecord]:
    """Return records carrying deterministic tile/product class labels."""
    by_tile = {
        str(key).casefold(): normalize_scene_class(value)
        for key, value in (category_by_tile or {}).items()
    }
    by_product = {
        str(key).casefold(): normalize_scene_class(value)
        for key, value in (category_by_product or {}).items()
    }
    annotated = []
    missing = []
    for record in records:
        scene_class = by_tile.get(record.tile_id.casefold())
        if scene_class is None:
            product = (record.sentinel_product or record.source_product).casefold()
            matches = [
                (prefix, value)
                for prefix, value in by_product.items()
                if product.startswith(prefix)
            ]
            if matches:
                scene_class = max(matches, key=lambda item: len(item[0]))[1]
        if scene_class is None and record.scene_class != "unlabeled":
            scene_class = normalize_scene_class(record.scene_class)
        if scene_class is None:
            scene_class = "unlabeled"
            missing.append(record.patch)
        annotated.append(replace(record, scene_class=scene_class))
    if require_all and missing:
        raise ValueError(
            f"{len(missing)} patches lack a scene class. Add CATEGORY_BY_TILE or "
            f"CATEGORY_BY_PRODUCT entries; first missing patch: {missing[0]}"
        )
    return annotated


def category_split_rows(records: list[ManifestRecord]) -> list[dict[str, object]]:
    counts = Counter(
        (record.scene_class, record.split)
        for record in records
        if record.split in ("train", "val", "test")
    )
    rows = []
    for scene_class in sorted({record.scene_class for record in records}):
        active = sum(counts[scene_class, split] for split in ("train", "val", "test"))
        if not active:
            continue
        rows.append({
            "scene_class": scene_class,
            "train": counts[scene_class, "train"],
            "val": counts[scene_class, "val"],
            "test": counts[scene_class, "test"],
            "active": active,
            "test_fraction": counts[scene_class, "test"] / active,
        })
    return rows


def validate_category_splits(
    records: list[ManifestRecord],
    *,
    minimum_test_fraction: float = 0.1,
    require_each_split: bool = True,
) -> list[dict[str, object]]:
    rows = category_split_rows(records)
    if not rows:
        raise ValueError("No active train/val/test records")
    problems = []
    for row in rows:
        if row["scene_class"] == "unlabeled":
            problems.append("unlabeled records remain")
        if require_each_split and any(not row[split] for split in ("train", "val", "test")):
            problems.append(f"{row['scene_class']} does not populate train, val and test")
        if row["test_fraction"] < minimum_test_fraction:
            problems.append(
                f"{row['scene_class']} test fraction {row['test_fraction']:.3f} is below "
                f"{minimum_test_fraction:.3f}"
            )
    if problems:
        raise ValueError("; ".join(problems))
    return rows


def export_portable_dataset(
    records: list[ManifestRecord],
    output_root: str | Path,
    *,
    manifest_name: str = "manifest.jsonl",
) -> Path:
    """Copy each pair NPZ once into split/class folders and write relative paths."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()

    exported = []
    for record in records:
        if record.split not in ("train", "val", "test"):
            continue
        source = Path(record.patch)
        if not source.is_file():
            raise FileNotFoundError(source)
        pair_id = source.parent.name
        relative = Path(record.split) / record.scene_class / record.tile_id / pair_id / source.name
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if (
                destination.stat().st_size != source.stat().st_size
                or digest(destination) != digest(source)
            ):
                raise ValueError(f"Existing exported patch differs: {destination}")
        else:
            shutil.copy2(source, destination)
        exported.append(replace(record, patch=relative.as_posix()))
    manifest = output_root / manifest_name
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in exported:
            handle.write(record.to_json() + "\n")
    temporary.replace(manifest)
    return manifest


def write_dataset_card(
    output_root: str | Path,
    records: list[ManifestRecord],
    *,
    preparation: dict[str, object],
) -> Path:
    output_root = Path(output_root)
    rows = category_split_rows(records)
    card = {
        "format": "geodiff-paired-npz-v1",
        "task": "real Landsat 30 m to Sentinel-2 10 m super-resolution",
        "scale": 3,
        "split_unit": "spatially isolated regions inside each MGRS tile",
        "warning": "This protocol tests held-out regions, not unseen-tile transfer.",
        "preparation": preparation,
        "categories": rows,
    }
    path = output_root / "dataset_card.json"
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return path
