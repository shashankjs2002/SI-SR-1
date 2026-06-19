"""Common benchmark harness for external super-resolution backbones."""

from .models import MODEL_SPECS, build_benchmark_model

__all__ = ["MODEL_SPECS", "build_benchmark_model"]
