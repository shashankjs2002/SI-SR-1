from __future__ import annotations

import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from geodiff_gan.metrics import OptionalMetricSuite, _load_dists_model


class _FakeDISTS(nn.Module):
    def __init__(self, load_weights: bool = True) -> None:
        super().__init__()
        if load_weights:
            raise AssertionError("the package's hard-coded weight loader must not run")
        self.alpha = nn.Parameter(torch.zeros(1, 2, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, 2, 1, 1))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).abs().mean((1, 2, 3))


class _InputRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.inputs = (prediction.detach().clone(), target.detach().clone())
        return prediction.new_tensor([0.25])


class OptionalMetricTests(unittest.TestCase):
    def test_dists_loads_weights_from_installed_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package_dir = Path(directory) / "DISTS_pytorch"
            package_dir.mkdir()
            package_file = package_dir / "__init__.py"
            package_file.touch()
            expected_alpha = torch.full((1, 2, 1, 1), 0.2)
            expected_beta = torch.full((1, 2, 1, 1), 0.3)
            torch.save(
                {"alpha": expected_alpha, "beta": expected_beta},
                package_dir / "weights.pt",
            )
            package = types.ModuleType("DISTS_pytorch")
            package.__file__ = str(package_file)
            package.DISTS = _FakeDISTS

            with mock.patch.dict(sys.modules, {"DISTS_pytorch": package}):
                model = _load_dists_model(torch.device("cpu"))

            self.assertTrue(torch.equal(model.alpha, expected_alpha))
            self.assertTrue(torch.equal(model.beta, expected_beta))

    def test_optional_dependency_failure_does_not_abort_suite(self) -> None:
        broken_lpips = types.ModuleType("lpips")

        def fail_lpips(*args, **kwargs):
            raise RuntimeError("LPIPS test failure")

        broken_lpips.LPIPS = fail_lpips
        missing_dists = types.ModuleType("DISTS_pytorch")
        missing_dists.__file__ = str(Path("missing") / "__init__.py")
        missing_dists.DISTS = _FakeDISTS
        with (
            mock.patch.dict(
                sys.modules,
                {"lpips": broken_lpips, "DISTS_pytorch": missing_dists},
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            suite = OptionalMetricSuite(torch.device("cpu"), enabled=True)

        self.assertEqual(suite.available_metrics, ())
        self.assertEqual(set(suite.load_errors), {"lpips", "dists"})
        self.assertEqual(len(caught), 2)

    def test_lpips_and_dists_receive_their_required_input_ranges(self) -> None:
        suite = OptionalMetricSuite(torch.device("cpu"), enabled=False)
        lpips_model = _InputRecorder()
        dists_model = _InputRecorder()
        suite.lpips_model = lpips_model
        suite.dists_model = dists_model
        prediction = torch.tensor([[[[0.0, 1.0]]]]).expand(1, 3, 1, 2)
        target = torch.full_like(prediction, 0.5)

        values = suite(prediction, target)

        assert lpips_model.inputs is not None
        assert dists_model.inputs is not None
        self.assertEqual(float(lpips_model.inputs[0].min()), -1.0)
        self.assertEqual(float(lpips_model.inputs[0].max()), 1.0)
        self.assertEqual(float(dists_model.inputs[0].min()), 0.0)
        self.assertEqual(float(dists_model.inputs[0].max()), 1.0)
        self.assertEqual(set(values), {"lpips", "dists"})


if __name__ == "__main__":
    unittest.main()
