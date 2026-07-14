from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved PINN checkpoints")
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.checkpoints:
        data = torch.load(path, map_location="cpu", weights_only=False)
        rows.append({"checkpoint": str(path), "method": data["method"], "problem": data["problem"], **data["metrics"], "eta": data["indicator"]["eta"]})
    rendered = json.dumps(rows, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
