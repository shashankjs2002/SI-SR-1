from __future__ import annotations

import argparse

import torch

from ..config import load_config
from ..training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GeoDiff-GAN stage")
    parser.add_argument("--config", required=True)
    parser.add_argument("--defaults")
    args = parser.parse_args()
    try:
        Trainer(load_config(args.config, args.defaults)).train()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

