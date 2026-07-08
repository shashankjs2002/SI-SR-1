from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelSpec:
    name: str
    paper: str
    year: int
    venue: str
    repository: str
    directory: str
    remote_sensing_specific: bool
    optional_dependency: str | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "swinir": ModelSpec(
        "SwinIR",
        "SwinIR: Image Restoration Using Swin Transformer",
        2021,
        "ICCV Workshops",
        "https://github.com/JingyunLiang/SwinIR.git",
        "SwinIR",
        False,
    ),
    "hat": ModelSpec(
        "HAT-S",
        "Activating More Pixels in Image Super-Resolution Transformer",
        2023,
        "CVPR",
        "https://github.com/XPixelGroup/HAT.git",
        "HAT",
        False,
    ),
    "srformer": ModelSpec(
        "SRFormer",
        "SRFormer: Permuted Self-Attention for Single Image Super-Resolution",
        2023,
        "ICCV",
        "https://github.com/HVision-NKU/SRFormer.git",
        "SRFormer",
        False,
    ),
    "dat": ModelSpec(
        "DAT",
        "Dual Aggregation Transformer for Image Super-Resolution",
        2023,
        "ICCV",
        "https://github.com/zhengchen1999/DAT.git",
        "DAT",
        False,
    ),
    "omnisr": ModelSpec(
        "OmniSR",
        "Omni Aggregation Networks for Lightweight Image Super-Resolution",
        2023,
        "CVPR",
        "https://github.com/Francis0625/Omni-SR.git",
        "OmniSR",
        False,
    ),
    "mfghmoe": ModelSpec(
        "MFG-HMoE",
        "Heterogeneous Mixture of Experts for Remote Sensing Image Super-Resolution",
        2025,
        "IEEE GRSL",
        "https://github.com/Mr-Bamboo/MFG-HMoE.git",
        "MFG-HMoE",
        True,
    ),
    "ttst": ModelSpec(
        "TTST",
        "TTST: A Top-k Token Selective Transformer for Remote Sensing Image Super-Resolution",
        2024,
        "IEEE TIP",
        "https://github.com/XY-boy/TTST.git",
        "TTST",
        True,
    ),
    "fremamba": ModelSpec(
        "FMSR/FreMamba",
        "Frequency-Assisted Mamba for Remote Sensing Image Super-Resolution",
        2024,
        "IEEE TMM",
        "https://github.com/XY-boy/FreMamba.git",
        "FreMamba",
        True,
        "mamba_ssm",
    ),
    "swin2mose": ModelSpec(
        "Swin2-MoSE",
        "Swin2-MoSE: A New Single Image Super-Resolution Model for Remote Sensing",
        2024,
        "arXiv",
        "https://github.com/IMPLabUniPr/swin2-mose.git",
        "Swin2-MoSE",
        True,
    ),
    "atd": ModelSpec(
        "ATD",
        "Transcending the Limit of Local Window: Advanced Super-Resolution Transformer with Adaptive Token Dictionary",
        2024,
        "CVPR",
        "https://github.com/CVL-UESTC/Adaptive-Token-Dictionary.git",
        "ATD",
        False,
    ),
    "mambair": ModelSpec(
        "MambaIR",
        "MambaIR: A Simple Baseline for Image Restoration with State-Space Model",
        2024,
        "ECCV",
        "https://github.com/csguoh/MambaIR.git",
        "MambaIR",
        False,
        "mamba_ssm",
    ),
    "mambairv2": ModelSpec(
        "MambaIRv2",
        "MambaIRv2: Attentive State Space Restoration",
        2025,
        "CVPR",
        "https://github.com/csguoh/MambaIR.git",
        "MambaIR",
        False,
        "mamba_ssm",
    ),
}


class _Registry:
    def register(self, obj: Any | None = None, **_: Any):
        def decorator(value: Any) -> Any:
            return value

        return decorator(obj) if obj is not None else decorator


def _install_basicsr_stubs() -> None:
    if "basicsr.utils.registry" in sys.modules:
        return
    basicsr = types.ModuleType("basicsr")
    basicsr.__path__ = []  # type: ignore[attr-defined]
    utils = types.ModuleType("basicsr.utils")
    utils.__path__ = []  # type: ignore[attr-defined]
    registry = types.ModuleType("basicsr.utils.registry")
    registry.ARCH_REGISTRY = _Registry()
    archs = types.ModuleType("basicsr.archs")
    archs.__path__ = []  # type: ignore[attr-defined]
    arch_util = types.ModuleType("basicsr.archs.arch_util")
    arch_util.to_2tuple = lambda value: value if isinstance(value, tuple) else (value, value)
    arch_util.trunc_normal_ = torch.nn.init.trunc_normal_
    sys.modules.update(
        {
            "basicsr": basicsr,
            "basicsr.utils": utils,
            "basicsr.utils.registry": registry,
            "basicsr.archs": archs,
            "basicsr.archs.arch_util": arch_util,
        }
    )


def _install_thop_stub() -> None:
    if "thop" in sys.modules:
        return
    module = types.ModuleType("thop")

    def unavailable(*_: Any, **__: Any) -> tuple[None, None]:
        raise RuntimeError("THOP profiling is not part of benchmark training")

    module.profile = unavailable
    sys.modules["thop"] = module


@contextmanager
def _source_path(path: Path) -> Iterator[None]:
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        if value in sys.path:
            sys.path.remove(value)


def _load_file(name: str, path: Path, package_path: Path | None = None) -> types.ModuleType:
    if package_path is not None:
        package_name = name.rsplit(".", 1)[0]
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ResizeConvHead(nn.Module):
    """Two-stage resize-convolution head with no sub-pixel rearrangement."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, scale: int = 4) -> None:
        super().__init__()
        if scale != 4:
            raise ValueError("The benchmark head currently supports x4 only")
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up1 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.up2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.hr = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.out = nn.Conv2d(hidden_channels, 3, 3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.pre(features)
        value = F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
        value = F.leaky_relu(self.up1(value), 0.2, inplace=True)
        value = F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
        value = F.leaky_relu(self.up2(value), 0.2, inplace=True)
        return self.out(F.leaky_relu(self.hr(value), 0.2, inplace=True))


class TransformerBackboneSR(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_channels: int,
        scale: int = 4,
        tuple_features: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.scale = scale
        self.tuple_features = tuple_features
        auxiliary_channels = 64 if tuple_features else 0
        self.head = ResizeConvHead(feature_channels + auxiliary_channels, scale=scale)

    def _pad(self, image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height, width = image.shape[-2:]
        if hasattr(self.backbone, "check_image_size"):
            return self.backbone.check_image_size(image), height, width
        window = int(getattr(self.backbone, "window_size", 1))
        pad_h = (window - height % window) % window
        pad_w = (window - width % window) % window
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        return image, height, width

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        padded, height, width = self._pad(image)
        mean = getattr(self.backbone, "mean", None)
        image_range = float(getattr(self.backbone, "img_range", 1.0))
        normalized = padded
        if mean is not None:
            normalized = normalized - mean.to(device=image.device, dtype=image.dtype)
        normalized = normalized * image_range
        shallow = self.backbone.conv_first(normalized)
        body = self.backbone.forward_features(shallow)
        auxiliary = None
        if isinstance(body, tuple):
            body, auxiliary = body
        features = self.backbone.conv_after_body(body) + shallow
        if auxiliary is not None:
            features = torch.cat((features, auxiliary), dim=1)
        residual = self.head(features)
        anchor = F.interpolate(
            padded,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )
        output = (anchor + residual).clamp(0, 1)
        return output[:, :, : height * self.scale, : width * self.scale]


class OmniBackboneSR(nn.Module):
    def __init__(self, backbone: nn.Module, scale: int = 4) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.up = nn.Identity()
        self.scale = scale
        self.head = ResizeConvHead(64, scale=scale)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        padded = self.backbone.check_image_size(image)
        shallow = self.backbone.input(padded)
        features = self.backbone.output(self.backbone.residual_layer(shallow)) + shallow
        residual = self.head(features)
        anchor = F.interpolate(
            padded,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )
        output = (anchor + residual).clamp(0, 1)
        return output[:, :, : height * self.scale, : width * self.scale]


class ResizeMoEHead(nn.Module):
    """No-PixelShuffle x4 head retaining MFG-HMoE dual-routing experts."""

    def __init__(self, hmoe_class: type[nn.Module], in_channels: int = 180) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.expert1 = hmoe_class(64, 64, 16)
        self.expert2 = hmoe_class(64, 64, 16)
        self.up1 = nn.Conv2d(64, 64, 3, padding=1)
        self.up2 = nn.Conv2d(64, 64, 3, padding=1)
        self.out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, features: torch.Tensor, routing: torch.Tensor) -> torch.Tensor:
        value = self.pre(features)
        routing1 = F.interpolate(
            routing, size=value.shape[-2:], mode="bilinear", align_corners=False
        )
        value = self.expert1(value, routing1)
        value = F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
        value = F.leaky_relu(self.up1(value), 0.2, inplace=True)
        routing2 = F.interpolate(
            routing, size=value.shape[-2:], mode="bilinear", align_corners=False
        )
        value = self.expert2(value, routing2)
        value = F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
        value = F.leaky_relu(self.up2(value), 0.2, inplace=True)
        return self.out(value)


class MFGHMoEBackboneSR(nn.Module):
    def __init__(self, backbone: nn.Module, hmoe_class: type[nn.Module]) -> None:
        super().__init__()
        self.backbone = backbone
        self.scale = 4
        self.head = ResizeMoEHead(hmoe_class)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        window = int(self.backbone.window_size)
        pad_h = (window - height % window) % window
        pad_w = (window - width % window) % window
        padded = (
            F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
            if pad_h or pad_w
            else image
        )
        shallow = self.backbone.conv_first(padded)
        body, routing = self.backbone.forward_features(shallow)
        features = self.backbone.conv_after_body(body) + shallow
        residual = self.head(features, routing)
        anchor = F.interpolate(
            padded, scale_factor=4, mode="bicubic", align_corners=False
        )
        output = (anchor + residual).clamp(0, 1)
        return output[:, :, : height * 4, : width * 4]


class FirstTensorOutput(nn.Module):
    """Compatibility wrapper for official models that return auxiliary losses."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image)
        if isinstance(output, tuple):
            return output[0]
        return output


def _build_swinir(root: Path) -> nn.Module:
    module = _load_file("official_swinir", root / "models" / "network_swinir.py")
    backbone = module.SwinIR(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=8,
        mlp_ratio=2,
        upscale=4,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        use_checkpoint=False,
    )
    return TransformerBackboneSR(backbone, 180)


def _build_hat(root: Path) -> nn.Module:
    _install_basicsr_stubs()
    module = _load_file("official_hat_arch", root / "hat" / "archs" / "hat_arch.py")
    backbone = module.HAT(
        img_size=64,
        in_chans=3,
        embed_dim=144,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=16,
        compress_ratio=24,
        squeeze_factor=24,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=2,
        upscale=4,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        use_checkpoint=False,
    )
    return TransformerBackboneSR(backbone, 144)


def _build_srformer(root: Path) -> nn.Module:
    _install_basicsr_stubs()
    arch_path = root / "basicsr" / "archs"
    arch_util = types.ModuleType("official_srformer.arch_util")
    arch_util.to_2tuple = lambda value: value if isinstance(value, tuple) else (value, value)
    arch_util.trunc_normal_ = torch.nn.init.trunc_normal_
    sys.modules["official_srformer.arch_util"] = arch_util
    module = _load_file(
        "official_srformer.srformer_arch",
        arch_path / "srformer_arch.py",
        package_path=arch_path,
    )
    backbone = module.SRFormer(
        img_size=48,
        in_chans=3,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=24,
        mlp_ratio=2,
        upscale=4,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        use_checkpoint=False,
    )
    return TransformerBackboneSR(backbone, 180)


def _build_dat(root: Path) -> nn.Module:
    _install_basicsr_stubs()
    module = _load_file("official_dat_arch", root / "basicsr" / "archs" / "dat_arch.py")
    backbone = module.DAT(
        img_size=64,
        in_chans=3,
        embed_dim=180,
        depth=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        split_size=(8, 32),
        expansion_factor=4,
        upscale=4,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        use_chk=False,
    )
    return TransformerBackboneSR(backbone, 180)


def _build_omnisr(root: Path) -> nn.Module:
    with _source_path(root):
        module = _load_file("official_omnisr", root / "components" / "OmniSR.py")
        backbone = module.OmniSR(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            upsampling=4,
            res_num=5,
            block_num=1,
            bias=True,
            window_size=8,
            pe=True,
            ffn_bias=True,
            block_script_name="OSA",
            block_class_name="OSA_Block",
        )
    return OmniBackboneSR(backbone)


def _build_mfghmoe(root: Path) -> nn.Module:
    _install_basicsr_stubs()
    module = _load_file(
        "official_mfghmoe_arch",
        root / "hmoe" / "archs" / "mfghmoe_arch.py",
    )
    backbone = module.MFGHMOE(
        img_size=64,
        in_chans=3,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=16,
        mlp_ratio=2,
        upscale=4,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        use_checkpoint=False,
    )
    return MFGHMoEBackboneSR(backbone, module.HMoE)


def _build_ttst(root: Path) -> nn.Module:
    _install_basicsr_stubs()
    _install_thop_stub()
    module = _load_file("official_ttst_arch", root / "model_archs" / "TTST_arc.py")
    backbone = module.TTST(
        img_size=64,
        in_chans=3,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=8,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=2,
        upscale=4,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        use_checkpoint=False,
    )
    return TransformerBackboneSR(backbone, 180)


def _build_fremamba(root: Path) -> nn.Module:
    _install_basicsr_stubs()
    module = _load_file(
        "official_fremamba_arch",
        root / "model_archs" / "fremamba.py",
    )
    backbone = module.FreMamba(
        img_size=64,
        in_chans=3,
        embed_dim=96,
        depths=(6, 6, 6, 6, 6, 6),
        d_state=16,
        mlp_ratio=2,
        upscale=4,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        use_checkpoint=False,
    )
    return TransformerBackboneSR(backbone, 96)


RESIZE_CONV_BUILDERS = {
    "swinir": _build_swinir,
    "hat": _build_hat,
    "srformer": _build_srformer,
    "dat": _build_dat,
    "omnisr": _build_omnisr,
    "mfghmoe": _build_mfghmoe,
    "ttst": _build_ttst,
    "fremamba": _build_fremamba,
}


def _build_official_swinir(root: Path) -> nn.Module:
    module = _load_file("official_swinir_faithful", root / "models" / "network_swinir.py")
    return module.SwinIR(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=8,
        mlp_ratio=2,
        upscale=4,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        use_checkpoint=False,
    )


def _build_official_hat(root: Path) -> nn.Module:
    _install_basicsr_stubs()
    module = _load_file(
        "official_hat_arch_faithful",
        root / "hat" / "archs" / "hat_arch.py",
    )
    return module.HAT(
        img_size=64,
        in_chans=3,
        embed_dim=144,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=16,
        compress_ratio=24,
        squeeze_factor=24,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=2,
        upscale=4,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        use_checkpoint=False,
    )


def _build_official_srformer(root: Path) -> nn.Module:
    _install_basicsr_stubs()
    arch_path = root / "basicsr" / "archs"
    arch_util = types.ModuleType("official_srformer_faithful.arch_util")
    arch_util.to_2tuple = lambda value: value if isinstance(value, tuple) else (value, value)
    arch_util.trunc_normal_ = torch.nn.init.trunc_normal_
    sys.modules["official_srformer_faithful.arch_util"] = arch_util
    module = _load_file(
        "official_srformer_faithful.srformer_arch",
        arch_path / "srformer_arch.py",
        package_path=arch_path,
    )
    return module.SRFormer(
        img_size=48,
        in_chans=3,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=24,
        mlp_ratio=2,
        upscale=4,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        use_checkpoint=False,
    )


def _build_official_dat(root: Path) -> nn.Module:
    _install_basicsr_stubs()
    module = _load_file(
        "official_dat_arch_faithful",
        root / "basicsr" / "archs" / "dat_arch.py",
    )
    return module.DAT(
        img_size=64,
        in_chans=3,
        embed_dim=180,
        depth=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        split_size=(8, 32),
        expansion_factor=4,
        upscale=4,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        use_chk=False,
    )


def _build_official_omnisr(root: Path) -> nn.Module:
    with _source_path(root):
        module = _load_file(
            "official_omnisr_faithful",
            root / "components" / "OmniSR.py",
        )
        return module.OmniSR(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            upsampling=4,
            res_num=5,
            block_num=1,
            bias=True,
            window_size=8,
            pe=True,
            ffn_bias=True,
            block_script_name="OSA",
            block_class_name="OSA_Block",
        )


def _build_official_mfghmoe(root: Path) -> nn.Module:
    _install_basicsr_stubs()
    module = _load_file(
        "official_mfghmoe_arch_faithful",
        root / "hmoe" / "archs" / "mfghmoe_arch.py",
    )
    return module.MFGHMOE(
        img_size=64,
        in_chans=3,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=16,
        mlp_ratio=2,
        upscale=4,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        use_checkpoint=False,
    )


def _build_official_ttst(root: Path) -> nn.Module:
    _install_basicsr_stubs()
    _install_thop_stub()
    module = _load_file(
        "official_ttst_arch_faithful",
        root / "model_archs" / "TTST_arc.py",
    )
    return module.TTST(
        img_size=64,
        in_chans=3,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=8,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=2,
        upscale=4,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        use_checkpoint=False,
    )


def _build_official_fremamba(root: Path) -> nn.Module:
    _install_basicsr_stubs()
    module = _load_file(
        "official_fremamba_arch_faithful",
        root / "model_archs" / "fremamba.py",
    )
    return module.FreMamba(
        img_size=64,
        in_chans=3,
        embed_dim=96,
        depths=(6, 6, 6, 6, 6, 6),
        d_state=16,
        mlp_ratio=2,
        upscale=4,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        use_checkpoint=False,
    )


def _build_official_swin2mose(root: Path) -> nn.Module:
    module = _load_file(
        "official_swin2mose_faithful",
        root / "swin2_mose_model" / "model.py",
    )
    return FirstTensorOutput(
        module.Swin2MoSE(
            img_size=64,
            patch_size=1,
            in_chans=3,
            embed_dim=180,
            depths=(6, 6, 6, 6, 6, 6),
            num_heads=(6, 6, 6, 6, 6, 6),
            window_size=8,
            mlp_ratio=2,
            upscale=4,
            img_range=1.0,
            upsampler="pixelshuffle",
            resi_connection="1conv",
            use_checkpoint=False,
            MoE_config={
                "num_experts": 4,
                "k": 1,
                "noisy_gating": True,
                "input_size": 180,
            },
        )
    )


def _build_official_atd(root: Path) -> nn.Module:
    _install_basicsr_stubs()
    module = _load_file(
        "official_atd_arch_faithful",
        root / "basicsr" / "archs" / "atd_arch.py",
    )
    return module.ATD(
        upscale=4,
        img_size=64,
        in_chans=3,
        embed_dim=216,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(4, 4, 4, 4, 4, 4),
        window_size=16,
        dim_ffn_td=16,
        category_size=256,
        num_tokens=512,
        reducted_dim=16,
        convffn_kernel_size=5,
        img_range=1.0,
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        use_checkpoint=False,
    )


def _build_official_mambair(root: Path) -> nn.Module:
    _install_basicsr_stubs()
    module = _load_file(
        "official_mambair_arch_faithful",
        root / "basicsr" / "archs" / "mambair_arch.py",
    )
    return module.MambaIR(
        upscale=4,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        d_state=16,
        mlp_ratio=2,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        use_checkpoint=False,
    )


def _build_official_mambairv2(root: Path) -> nn.Module:
    _install_basicsr_stubs()
    module = _load_file(
        "official_mambairv2_arch_faithful",
        root / "basicsr" / "archs" / "mambairv2_arch.py",
    )
    return module.MambaIRv2(
        upscale=4,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        d_state=16,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=16,
        inner_rank=64,
        num_tokens=128,
        convffn_kernel_size=5,
        img_range=1.0,
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        use_checkpoint=False,
    )


OFFICIAL_BUILDERS = {
    "swinir": _build_official_swinir,
    "hat": _build_official_hat,
    "srformer": _build_official_srformer,
    "dat": _build_official_dat,
    "omnisr": _build_official_omnisr,
    "mfghmoe": _build_official_mfghmoe,
    "ttst": _build_official_ttst,
    "fremamba": _build_official_fremamba,
    "swin2mose": _build_official_swin2mose,
    "atd": _build_official_atd,
    "mambair": _build_official_mambair,
    "mambairv2": _build_official_mambairv2,
}


def pixelshuffle_modules(model: nn.Module) -> list[str]:
    return [
        name for name, module in model.named_modules() if isinstance(module, nn.PixelShuffle)
    ]


def build_benchmark_model(
    name: str,
    source_root: str | Path,
    architecture_mode: str = "official",
) -> nn.Module:
    key = name.lower()
    if key not in MODEL_SPECS:
        raise KeyError(f"Unknown model {name!r}; choose from {sorted(MODEL_SPECS)}")
    if architecture_mode not in ("official", "resize_conv_ablation"):
        raise ValueError(
            "architecture_mode must be 'official' or 'resize_conv_ablation'"
        )
    root = Path(source_root) / MODEL_SPECS[key].directory
    if not root.exists():
        raise FileNotFoundError(
            f"Official source missing at {root}. Clone {MODEL_SPECS[key].repository} first."
        )
    if architecture_mode == "official":
        return OFFICIAL_BUILDERS[key](root)
    if key not in RESIZE_CONV_BUILDERS:
        raise ValueError(
            f"{MODEL_SPECS[key].name} does not define a resize-convolution ablation. "
            "Use architecture_mode='official' for faithful published-model comparisons."
        )
    model = RESIZE_CONV_BUILDERS[key](root)
    offenders = pixelshuffle_modules(model)
    if offenders:
        raise RuntimeError(
            f"PixelShuffle remains in resize-convolution ablation: {offenders}"
        )
    return model
