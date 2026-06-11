from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
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


def load_manifest(path: str | Path, split: str | None = None) -> list[ManifestRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = ManifestRecord(**json.loads(line))
            if split is None or record.split == split:
                records.append(record)
    return records


def write_manifest(path: str | Path, records: list[ManifestRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
