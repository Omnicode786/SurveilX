"""Evaluate existing labeled tracking JSON without training, inference or downloads."""
import argparse
import hashlib
import json
from pathlib import Path

from training.tracking_evaluation import evaluate_tracking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iou", type=float, default=.5)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.input.resolve():
        raise ValueError("Preserve existing reports; choose a new output path")
    data = args.input.read_bytes()
    report = evaluate_tracking(json.loads(data), args.iou)
    report["input_sha256"] = hashlib.sha256(data).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
