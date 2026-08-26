from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from geodiff_gan.data.manifest import load_manifest
from geodiff_gan.losses import charbonnier, ssim
from geodiff_gan.metrics import edge_f1, psnr, remote_sensing_metrics


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def channel_first(value: np.ndarray, key: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(value)).float()
    if tensor.ndim != 3:
        raise ValueError(f"{key} must be three-dimensional, found {tensor.shape}")
    if tensor.shape[0] <= 16:
        return tensor
    if tensor.shape[-1] <= 16:
        return tensor.permute(2, 0, 1)
    raise ValueError(f"Cannot determine channel dimension for {key}: {tensor.shape}")


def load_asddpm(
    repository: Path,
    checkpoint_path: Path,
    rrdb_checkpoint_path: Path,
    device: torch.device,
    scale: int,
):
    repository = repository.resolve()
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))

    original_argv = sys.argv[:]
    dataset_name = "OLI2MSI" if scale == 3 else "ALSAT"
    lr_encoder = f"rrdb{scale}"
    patch_size = 96 if scale == 3 else 128
    sys.argv = [
        "asddpm_pretrained_eval",
        "--lr_encoder",
        lr_encoder,
        "--diffusion_net",
        "unetdualfusion",
        "--data_train",
        dataset_name,
        "--data_train_dir",
        dataset_name,
        "--scale",
        str(scale),
        "--patch_size",
        str(patch_size),
        "--lr",
        "0.0001",
    ]
    try:
        with working_directory(repository):
            from utils.hparams import hparams, set_hparams

            set_hparams(
                config=str(repository / "configs" / "diffsr_alsat4x.yaml"),
                exp_name="",
                hparams_str="",
                print_hparams=False,
            )
            from models.DiffusionNet.Unetdualfusion import UnetDualFusion
            from models.diffusion import GaussianDiffusion

            if scale == 3:
                from models.LREncoder.RRDB.rrdb3 import RRDBNet3 as RRDBNet
            else:
                from models.LREncoder.RRDB.rrdb4 import RRDBNet4 as RRDBNet

            dimensions = [int(value) for value in hparams["unet_dim_mults"].split("|")]
            denoiser = UnetDualFusion(
                int(hparams["hidden_size"]),
                out_dim=3,
                cond_dim=int(hparams["rrdb_num_feat"]),
                dim_mults=dimensions,
            )
            rrdb = RRDBNet(
                3,
                3,
                int(hparams["rrdb_num_feat"]),
                int(hparams["rrdb_num_block"]),
                int(hparams["rrdb_num_feat"]) // 2,
            )
            rrdb_state = torch.load(
                rrdb_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if isinstance(rrdb_state, dict) and "state_dict" in rrdb_state:
                rrdb_state = rrdb_state["state_dict"]
            rrdb_loaded_keys = set(rrdb.state_dict()).intersection(rrdb_state)
            if not rrdb_loaded_keys:
                raise RuntimeError(
                    f"RRDB checkpoint {rrdb_checkpoint_path} does not match {lr_encoder}"
                )
            rrdb.load_state_dict(rrdb_state, strict=False)
            model = GaussianDiffusion(
                denoise_fn=denoiser,
                rrdb_net=rrdb,
                timesteps=int(hparams["timesteps"]),
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            state = checkpoint.get("state_dict", {}).get("model", checkpoint)
            model_keys = set(model.state_dict())
            loaded_keys = model_keys.intersection(state)
            if not loaded_keys:
                raise RuntimeError(
                    f"Checkpoint {checkpoint_path} has no keys matching the official "
                    "ASDDPM architecture"
                )
            incompatible = model.load_state_dict(state, strict=False)
            model = model.to(device).eval()
            model.sample_tqdm = False
    finally:
        sys.argv = original_argv

    print(
        "[ASDDPM] checkpoint load:",
        {
            "global_step": checkpoint.get("global_step"),
            "missing_keys": len(incompatible.missing_keys),
            "unexpected_keys": len(incompatible.unexpected_keys),
            "timesteps": model.num_timesteps,
            "scale": scale,
            "lr_encoder": lr_encoder,
            "loaded_parameter_tensors": len(loaded_keys),
            "model_parameter_tensors": len(model_keys),
            "rrdb_loaded_parameter_tensors": len(rrdb_loaded_keys),
        },
        flush=True,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[ASDDPM] missing:", incompatible.missing_keys[:20], flush=True)
        print("[ASDDPM] unexpected:", incompatible.unexpected_keys[:20], flush=True)
    return model


def official_bicubic(lr: torch.Tensor, scale: int, repository: Path) -> torch.Tensor:
    with working_directory(repository):
        from data.imgproc import imresize

        array = lr.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        resized = imresize(array, float(scale))
    return channel_first(resized, "official_bicubic")


def calculate_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: int,
) -> dict[str, float]:
    prediction = prediction[None].float()
    target = target[None].float()
    mask = mask[None].float()
    values = {
        "l1": float(charbonnier(prediction, target, epsilon=0.0, mask=mask)),
        "psnr": float(psnr(prediction, target, mask=mask)),
        "ssim": float(ssim(prediction, target, mask=mask)),
        "edge_f1": float(edge_f1(prediction, target, mask=mask)),
    }
    values.update(remote_sensing_metrics(prediction, target, scale=scale, mask=mask))
    quantized_prediction = prediction.mul(255).round().div(255)
    quantized_target = target.mul(255).round().div(255)
    values["psnr_uint8"] = float(psnr(quantized_prediction, quantized_target, mask=mask))
    values["ssim_uint8"] = float(ssim(quantized_prediction, quantized_target, mask=mask))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the unchanged pretrained ASDDPM on GeoDiff patches"
    )
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rrdb-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--scale", type=int, choices=(3, 4), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    if args.samples < 1:
        raise ValueError("--samples must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    repository = Path(args.official_repo).resolve()
    output = Path(args.output).resolve()
    prediction_root = output / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    records = load_manifest(args.manifest, split=args.split)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError(f"No records found for split {args.split!r}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = load_asddpm(
        repository,
        Path(args.checkpoint).resolve(),
        Path(args.rrdb_checkpoint).resolve(),
        device,
        args.scale,
    )
    rows: list[dict[str, object]] = []
    totals: defaultdict[str, float] = defaultdict(float)
    started = time.monotonic()

    for index, record in enumerate(
        tqdm(records, desc="pretrained ASDDPM", unit="patch")
    ):
        cache_path = prediction_root / (
            f"{index:05d}_{Path(record.patch).stem}_"
            f"n{args.samples}_seed{args.seed}.npz"
        )
        with np.load(record.patch) as patch:
            lr = channel_first(patch["lr"], "lr")[:3].clamp(0, 1)
            hr = channel_first(patch["hr"], "hr")[:3].clamp(0, 1)
            valid = channel_first(
                patch["valid_mask_hr"]
                if "valid_mask_hr" in patch.files
                else np.ones((1, *hr.shape[-2:]), dtype=np.float32),
                "valid_mask_hr",
            )[:1]
        expected = (lr.shape[-2] * args.scale, lr.shape[-1] * args.scale)
        if tuple(hr.shape[-2:]) != expected:
            raise ValueError(
                f"ASDDPM requires exact 3x geometry: LR={lr.shape}, HR={hr.shape}"
            )

        if cache_path.exists():
            with np.load(cache_path) as cache:
                mean = channel_first(cache["mean"], "mean").clamp(0, 1)
        else:
            lr_up = official_bicubic(lr, args.scale, repository).clamp(0, 1)
            generated = []
            for sample_index in range(args.samples):
                sample_seed = args.seed + index * 1009 + sample_index
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=args.amp and device.type == "cuda",
                ):
                    prediction, rrdb_output = model.sample(
                        lr[None].to(device),
                        lr_up[None].to(device),
                        (1, 3, *hr.shape[-2:]),
                    )
                generated.append(prediction[0].float().cpu().clamp(0, 1))
            stack = torch.stack(generated)
            mean = stack.mean(0)
            variance = stack.var(0, unbiased=False)
            np.savez_compressed(
                cache_path,
                mean=mean.numpy(),
                variance=variance.numpy(),
                source_patch=str(Path(record.patch).resolve()),
                dataset_index=index,
                samples=args.samples,
                timesteps=model.num_timesteps,
            )

        values = calculate_metrics(mean, hr, valid, scale=args.scale)
        row = {
            "dataset_index": index,
            "patch": str(Path(record.patch).resolve()),
            **values,
        }
        rows.append(row)
        for name, value in values.items():
            totals[name] += value

    summary = {
        name: total / len(rows) for name, total in totals.items()
    }
    summary.update(
        {
            "method": "pretrained_ASDDPM",
            "count": len(rows),
            "samples_per_patch": args.samples,
            "diffusion_steps": model.num_timesteps,
            "scale": args.scale,
            "dataset": "OLI2MSI" if args.scale == 3 else "ALSAT",
            "device": str(device),
            "amp": bool(args.amp and device.type == "cuda"),
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "rrdb_checkpoint": str(Path(args.rrdb_checkpoint).resolve()),
        }
    )
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output / "per_patch_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
