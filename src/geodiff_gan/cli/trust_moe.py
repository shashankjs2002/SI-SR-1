"""python -m geodiff_gan.cli.trust_moe train/evaluate --config ..."""
import argparse
import json

from ..experiments.trust_moe import evaluate, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--manifest")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--coverage", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.action == "train":
        if not args.config:
            parser.error("train needs --config")
        with open(args.config, encoding="utf-8") as stream:
            result = train(json.load(stream), device=args.device)
    else:
        if not args.checkpoint or not args.output:
            parser.error("evaluate needs --checkpoint and --output")
        result = evaluate(args.checkpoint, args.output, args.split, device=args.device,
                          coverage=args.coverage, top_k=args.top_k, manifest=args.manifest)
    print(result, flush=True)


if __name__ == "__main__":
    main()
