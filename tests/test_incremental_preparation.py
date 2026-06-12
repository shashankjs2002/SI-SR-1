from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from geodiff_gan.cli import prepare_sentinel
from geodiff_gan.data.manifest import ManifestRecord, load_manifest


class IncrementalPreparationTest(unittest.TestCase):
    def _arguments(self, root: Path) -> list[str]:
        return [
            "geodiff-prepare",
            "--input",
            str(root / "input"),
            "--output",
            str(root / "patches"),
            "--manifest",
            str(root / "manifest.jsonl"),
            "--state",
            str(root / "preparation-state.json"),
            "--max-products",
            "1",
            "--unmatched-split",
            "train",
        ]

    def test_existing_product_is_skipped_when_new_product_is_added(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            products = [root / "input" / "FIRST.SAFE"]

            def extract(product: Path, output: str, **_: object) -> list[ManifestRecord]:
                return [
                    ManifestRecord(
                        patch=str(Path(output) / f"{product.stem}.npz"),
                        tile_id=product.stem,
                        split="train",
                        row=0,
                        col=0,
                        valid_fraction=1.0,
                        source_product=product.name,
                    )
                ]

            with (
                mock.patch.object(
                    prepare_sentinel,
                    "discover_safe_products",
                    side_effect=lambda _: list(products),
                ),
                mock.patch.object(
                    prepare_sentinel,
                    "extract_product_patches",
                    side_effect=extract,
                ) as extractor,
            ):
                with mock.patch.object(sys, "argv", self._arguments(root)):
                    prepare_sentinel.main()
                products.append(root / "input" / "SECOND.SAFE")
                with mock.patch.object(sys, "argv", self._arguments(root)):
                    prepare_sentinel.main()

            self.assertEqual(
                [call.args[0].name for call in extractor.call_args_list],
                ["FIRST.SAFE", "SECOND.SAFE"],
            )
            records = load_manifest(root / "manifest.jsonl")
            self.assertEqual(
                [record.source_product for record in records],
                ["FIRST.SAFE", "SECOND.SAFE"],
            )
            state = json.loads(
                (root / "preparation-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                state["completed_products"],
                ["FIRST.SAFE", "SECOND.SAFE"],
            )

    def test_product_with_zero_valid_patches_is_still_remembered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            products = [root / "input" / "EMPTY.SAFE"]
            with (
                mock.patch.object(
                    prepare_sentinel,
                    "discover_safe_products",
                    return_value=products,
                ),
                mock.patch.object(
                    prepare_sentinel,
                    "extract_product_patches",
                    return_value=[],
                ) as extractor,
            ):
                with mock.patch.object(sys, "argv", self._arguments(root)):
                    prepare_sentinel.main()
                with mock.patch.object(sys, "argv", self._arguments(root)):
                    prepare_sentinel.main()

            self.assertEqual(extractor.call_count, 1)
            state = json.loads(
                (root / "preparation-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["completed_products"], ["EMPTY.SAFE"])

    def test_historical_rejected_manifest_prevents_reprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rejected = root / "rejected.jsonl"
            rejected.write_text(
                json.dumps(
                    {
                        "patch": "quarantine/old.npz",
                        "original_patch": "patches/old.npz",
                        "tile_id": "OLD",
                        "split": "train",
                        "row": 0,
                        "col": 0,
                        "valid_fraction": 1.0,
                        "source_product": "OLD.SAFE",
                        "filter": {"reason": "edge_black"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            products = [
                root / "input" / "OLD.SAFE",
                root / "input" / "NEW.SAFE",
            ]
            arguments = self._arguments(root) + [
                "--completed-manifest",
                str(rejected),
            ]

            def extract(product: Path, output: str, **_: object) -> list[ManifestRecord]:
                return [
                    ManifestRecord(
                        patch=str(Path(output) / f"{product.stem}.npz"),
                        tile_id=product.stem,
                        split="train",
                        row=0,
                        col=0,
                        valid_fraction=1.0,
                        source_product=product.name,
                    )
                ]

            with (
                mock.patch.object(
                    prepare_sentinel,
                    "discover_safe_products",
                    return_value=products,
                ),
                mock.patch.object(
                    prepare_sentinel,
                    "extract_product_patches",
                    side_effect=extract,
                ) as extractor,
                mock.patch.object(sys, "argv", arguments),
            ):
                prepare_sentinel.main()

            self.assertEqual(extractor.call_count, 1)
            self.assertEqual(extractor.call_args.args[0].name, "NEW.SAFE")

    def test_wrapper_and_inner_manifest_records_are_deduplicated(self) -> None:
        canonical = (
            "S2C_MSIL2A_20260527T050651_N0512_R019_"
            "T44RPQ_20260527T100616.SAFE"
        )
        wrapper_record = ManifestRecord(
            patch="patches/wrapper.npz",
            tile_id="44RPQ",
            split="train",
            row=0,
            col=0,
            valid_fraction=1.0,
            source_product=f"AYODHYA_{canonical}",
        )
        inner_record = ManifestRecord(
            patch="patches/inner.npz",
            tile_id="44RPQ",
            split="train",
            row=0,
            col=0,
            valid_fraction=1.0,
            source_product=canonical,
        )

        merged = prepare_sentinel._merge_records(
            [inner_record],
            [wrapper_record],
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0], wrapper_record)


if __name__ == "__main__":
    unittest.main()
