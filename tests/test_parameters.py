from __future__ import annotations

import unittest
from pathlib import Path

from geodiff_gan.config import load_config
from geodiff_gan.parameters import build_parameter_report, verify_parameter_report

ROOT = Path(__file__).resolve().parents[1]


class ParameterAuditTest(unittest.TestCase):
    def test_smoke_parameter_counts_and_stage_policy(self) -> None:
        config = load_config(
            ROOT / "configs/smoke.yaml",
            ROOT / "configs/default.yaml",
        )
        report = build_parameter_report(config, patches=702, world_size=1)
        verify_parameter_report(report)

        self.assertEqual(report["core_model"]["scalar_parameters"], 765_046)
        self.assertEqual(
            report["discriminators"]["scalar_parameters"],
            113_508,
        )
        self.assertEqual(
            report["training_stages"]["base"]["total_optimized_parameters"],
            26_211,
        )
        self.assertEqual(
            report["training_stages"]["joint"]["total_optimized_parameters"],
            697_140,
        )
        self.assertEqual(
            report["training_stages"]["edit"]["trainable_modules"],
            ["diffusion", "mapper", "decoder"],
        )
        self.assertEqual(
            report["training_configuration"]["optimizer_updates_per_epoch"],
            88,
        )


if __name__ == "__main__":
    unittest.main()
