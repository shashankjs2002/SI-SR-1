from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle/GeoDiff_GAN_OLI2MSI_FidelityTrust_V2_3x.ipynb"
CONFIG = ROOT / "configs/oli2msi_fidelity_trust_v2_3x.yaml"


class FidelityV2NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell["source"]) for cell in cls.notebook["cells"]
        )

    def test_notebook_is_clean_and_every_code_cell_compiles(self) -> None:
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs"), [])
            ast.parse("".join(cell["source"]))

    def test_numbered_sections_are_complete_and_ordered(self) -> None:
        sections = []
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            match = re.search(r"^##\s+(\d+)\.", "".join(cell["source"]), re.MULTILINE)
            if match:
                sections.append(int(match.group(1)))
        self.assertEqual(sections, list(range(29)))

    def test_locked_protocol_and_acceptance_gates_are_present(self) -> None:
        required = (
            'REPOSITORY_BRANCH = "SR-3x"',
            'EXPECTED_TRAIN, EXPECTED_TEST = 5225, 100',
            '"fidelity_rdn"',
            '"fidelity_swinir_v2"',
            '"joint_latent_source": "sampled"',
            '"joint_sample_steps": 20',
            '"trust_samples": 2',
            '"psnr_35"',
            '"gain_005"',
            '"ssim_non_degradation"',
            '"fraction_beating_base_psnr"',
            'evaluate("official_test_locked"',
            '"test", 4, 20, 1.0',
        )
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, self.source)

    def test_v2_configuration_uses_per_band_trust_without_pixelshuffle(self) -> None:
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        model = config["model"]
        self.assertEqual(model["trust_mode"], "per_band")
        self.assertFalse(model["use_uncertainty_abstention"])
        for key in (
            "base_upsample_mode",
            "vae_upsample_mode",
            "diffusion_upsample_mode",
            "decoder_upsample_mode",
        ):
            self.assertEqual(model[key], "resize_conv")
        self.assertEqual(config["data"]["input_mode"], "paired")
        self.assertEqual(model["scale"], 3)

    def test_default_profile_is_bounded_for_one_kaggle_session(self) -> None:
        required = (
            'EXECUTION_PROFILE = "kaggle_5h"',
            '"architecture_race": False',
            '"full_frame_epochs": 0',
            '"final_refit": False',
            '"base32": 105',
            '"diffusion": 60',
            'max_wall_time_minutes',
            'AUTO_DISCOVER_ATTACHED_RDN = True',
        )
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, self.source)


if __name__ == "__main__":
    unittest.main()
