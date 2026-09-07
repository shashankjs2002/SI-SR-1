from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path


@dataclass
class ManifestRecord:
    patch: str
    tile_id: str
    split: str
    row: int
    col: int
    valid_fraction: float
    source: str = "copernicus_sentinel2_l2a"
    license_id: str = "copernicus-free-full-open"
    caption: str = ""
    source_product: str = ""
    sentinel_product: str = ""
    landsat_product: str = ""
    sentinel_acquisition: str = ""
    landsat_acquisition: str = ""
    day_gap: int = 0
    scale: int = 4
    target_crs: str = ""
    target_transform: list[float] | None = None
    scene_class: str = "unlabeled"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True)


def deterministic_split(tile_id: str) -> str:
    bucket = int(hashlib.sha256(tile_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def split_tiles(tile_ids: list[str]) -> dict[str, str]:
    return {tile_id: deterministic_split(tile_id) for tile_id in sorted(set(tile_ids))}


def validate_tile_split_isolation(records: list[ManifestRecord]) -> None:
    assignments: dict[str, set[str]] = {}
    for record in records:
        assignments.setdefault(record.tile_id, set()).add(record.split)
    conflicts = {
        tile_id: sorted(splits)
        for tile_id, splits in assignments.items()
        if len(splits) > 1
    }
    if conflicts:
        details = ", ".join(
            f"{tile_id}={splits}" for tile_id, splits in sorted(conflicts.items())
        )
        raise ValueError(
            "MGRS tile leakage detected: one geographic tile was assigned to "
            f"multiple splits ({details}). Adjust the SAFE prefix rules."
        )


def validate_within_tile_spatial_isolation(
    records: list[ManifestRecord],
    patch_size: int,
) -> None:
    """Validate per-tile train/val/test coverage and non-overlapping footprints."""
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    grouped: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.tile_id].append(record)
    required = ("train", "val", "test")
    for tile_id, tile_records in sorted(grouped.items()):
        active = {
            split: [record for record in tile_records if record.split == split]
            for split in required
        }
        missing = [split for split, values in active.items() if not values]
        if missing:
            raise ValueError(
                f"Tile {tile_id!r} has no patches for split(s) {missing}. "
                "It needs greater spatial coverage for a leakage-safe within-tile split."
            )
        separated = False
        for coordinate in ("row", "col"):
            train_end = max(
                getattr(record, coordinate) + patch_size
                for record in active["train"]
            )
            val_start = min(
                getattr(record, coordinate) for record in active["val"]
            )
            val_end = max(
                getattr(record, coordinate) + patch_size
                for record in active["val"]
            )
            test_start = min(
                getattr(record, coordinate) for record in active["test"]
            )
            if train_end <= val_start and val_end <= test_start:
                separated = True
                break
        if not separated:
            raise ValueError(
                f"Tile {tile_id!r} does not have spatially isolated train/val/test "
                "footprints. Do not use random overlapping-patch splits."
            )


def assign_within_tile_spatial_splits(
    records: list[ManifestRecord],
    patch_size: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    minimum_validation_fraction: float = 0.0,
    minimum_test_fraction: float = 0.0,
    discard_split: str = "discard",
) -> dict[str, dict[str, object]]:
    """Assign every tile to leakage-safe spatial train/val/test regions.

    Windows crossing either region boundary are assigned to ``discard_split``.
    This guard band is necessary when extraction stride is smaller than the patch
    size, because adjacent patches otherwise share target pixels.
    """
    test_fraction = 1.0 - float(train_fraction) - float(validation_fraction)
    fractions = {
        "train": float(train_fraction),
        "val": float(validation_fraction),
        "test": test_fraction,
    }
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if any(value <= 0 for value in fractions.values()):
        raise ValueError(
            "train_fraction and validation_fraction must leave positive "
            "train, validation, and test fractions"
        )
    if not 0 <= minimum_validation_fraction < 1:
        raise ValueError("minimum_validation_fraction must be in [0, 1)")
    if not 0 <= minimum_test_fraction < 1:
        raise ValueError("minimum_test_fraction must be in [0, 1)")
    grouped: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.tile_id].append(record)
    reports: dict[str, dict[str, object]] = {}
    for tile_id, tile_records in sorted(grouped.items()):
        axis_spans = {
            axis: max(getattr(record, axis) for record in tile_records)
            - min(getattr(record, axis) for record in tile_records)
            for axis in ("row", "col")
        }
        axes = sorted(axis_spans, key=lambda axis: (-axis_spans[axis], axis))
        best: tuple[
            float, int, int, int, int, str, dict[int, str]
        ] | None = None
        for axis_rank, axis in enumerate(axes):
            starts = [getattr(record, axis) for record in tile_records]
            boundaries = sorted(
                set(starts) | {start + patch_size for start in starts}
            )
            for first_index, first_boundary in enumerate(boundaries[:-1]):
                for second_boundary in boundaries[first_index + 1 :]:
                    assignments: dict[int, str] = {}
                    counts: Counter[str] = Counter()
                    for record_index, (record, start) in enumerate(
                        zip(tile_records, starts)
                    ):
                        end = start + patch_size
                        if end <= first_boundary:
                            split = "train"
                        elif start >= first_boundary and end <= second_boundary:
                            split = "val"
                        elif start >= second_boundary:
                            split = "test"
                        else:
                            split = discard_split
                        assignments[record_index] = split
                        counts[split] += 1
                    if any(counts[split] == 0 for split in fractions):
                        continue
                    retained = sum(counts[split] for split in fractions)
                    if (
                        counts["val"] / retained < minimum_validation_fraction
                        or counts["test"] / retained < minimum_test_fraction
                    ):
                        continue
                    ratio_error = sum(
                        abs(counts[split] / retained - target)
                        for split, target in fractions.items()
                    )
                    discard_fraction = counts[discard_split] / len(tile_records)
                    score = ratio_error + 2.0 * discard_fraction
                    candidate = (
                        score,
                        counts[discard_split],
                        axis_rank,
                        first_boundary,
                        second_boundary,
                        axis,
                        assignments,
                    )
                    if best is None or candidate[:5] < best[:5]:
                        best = candidate
        if best is None:
            raise ValueError(
                f"Tile {tile_id!r} cannot provide leakage-safe train/val/test "
                f"regions for patch_size={patch_size}. Add more spatial coverage."
            )
        _, _, _, first_boundary, second_boundary, axis, assignments = best
        for record_index, split in assignments.items():
            tile_records[record_index].split = split
        counts = Counter(record.split for record in tile_records)
        retained = sum(counts[split] for split in fractions)
        reports[tile_id] = {
            "axis": axis,
            "boundaries": [first_boundary, second_boundary],
            "counts": dict(counts),
            "retained": retained,
            "discarded": counts[discard_split],
            "retained_fractions": {
                split: counts[split] / retained for split in fractions
            },
        }
    validate_within_tile_spatial_isolation(records, patch_size)
    return reports


def build_within_tile_spatial_folds(
    records: list[ManifestRecord],
    patch_size: int,
    folds: int = 5,
    discard_split: str = "discard",
) -> tuple[list[list[ManifestRecord]], dict[str, dict[str, object]]]:
    """Build leakage-safe spatial K-fold manifests for every represented tile."""
    if folds < 3:
        raise ValueError("folds must be at least 3 to separate train, val, and test")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    grouped: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.tile_id].append(record)
    region_by_patch: dict[str, int | None] = {}
    reports: dict[str, dict[str, object]] = {}
    for tile_id, tile_records in sorted(grouped.items()):
        spans = {
            axis: max(getattr(record, axis) for record in tile_records)
            + patch_size
            - min(getattr(record, axis) for record in tile_records)
            for axis in ("row", "col")
        }
        axis = sorted(spans, key=lambda value: (-spans[value], value))[0]
        minimum = min(getattr(record, axis) for record in tile_records)
        maximum = max(
            getattr(record, axis) + patch_size for record in tile_records
        )
        boundaries = [
            minimum + (maximum - minimum) * index / folds
            for index in range(folds + 1)
        ]
        region_counts: Counter[int | str] = Counter()
        for record in tile_records:
            start = getattr(record, axis)
            end = start + patch_size
            region: int | None = None
            for region_index, (left, right) in enumerate(
                zip(boundaries[:-1], boundaries[1:])
            ):
                if start >= left and end <= right:
                    region = region_index
                    break
            region_by_patch[record.patch] = region
            region_counts[region if region is not None else discard_split] += 1
        missing = [index for index in range(folds) if not region_counts[index]]
        if missing:
            raise ValueError(
                f"Tile {tile_id!r} cannot populate spatial fold region(s) {missing}. "
                "Use fewer folds or add more spatial coverage."
            )
        reports[tile_id] = {
            "axis": axis,
            "boundaries": boundaries,
            "region_counts": {
                str(index): region_counts[index] for index in range(folds)
            },
            "discarded": region_counts[discard_split],
        }
    manifests: list[list[ManifestRecord]] = []
    for fold in range(folds):
        test_region = fold
        validation_region = (fold + 1) % folds
        fold_records: list[ManifestRecord] = []
        for record in records:
            region = region_by_patch[record.patch]
            if region is None:
                split = discard_split
            elif region == test_region:
                split = "test"
            elif region == validation_region:
                split = "val"
            else:
                split = "train"
            fold_records.append(replace(record, split=split))
        manifests.append(fold_records)
    return manifests, reports


def load_manifest(
    path: str | Path,
    split: str | None = None,
    *,
    resolve_paths: bool = False,
) -> list[ManifestRecord]:
    manifest_path = Path(path)
    records = []
    record_fields = {field.name for field in fields(ManifestRecord)}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            record = ManifestRecord(
                **{key: item for key, item in value.items() if key in record_fields}
            )
            if resolve_paths and record.patch and not Path(record.patch).is_absolute():
                record.patch = str((manifest_path.parent / record.patch).resolve())
            if split is None or record.split == split:
                records.append(record)
    return records


def write_manifest(path: str | Path, records: list[ManifestRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
    temporary.replace(destination)
