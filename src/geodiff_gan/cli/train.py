from __future__ import annotations

import argparse

from ..config import load_config
from ..training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GeoDiff-GAN stage")
    parser.add_argument("--config", required=True)
    parser.add_argument("--defaults")
    args = parser.parse_args()
    Trainer(load_config(args.config, args.defaults)).train()


if __name__ == "__main__":
    main()

