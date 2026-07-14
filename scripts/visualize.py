from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import DirectPINN, IndirectPINN
from src.plotting import plot_adjoint, plot_controls, plot_histories, plot_metrics, plot_residuals, plot_state_comparison, plot_state_slices
from src.problems import make_problem


def load_checkpoint(path: Path, expected_method: str, device: str):
    data = torch.load(path, map_location=device, weights_only=False)
    if data["method"] != expected_method:
        raise ValueError(f"{path} contains method={data['method']}, expected {expected_method}")
    if data.get("control_parameterization") != "shared_U_psi(x,t)":
        raise ValueError(
            f"{path} uses the legacy two-network boundary control. "
            "Retrain it with the current shared U_psi(x,t) architecture before visualization."
        )
    model = DirectPINN() if expected_method == "direct" else IndirectPINN()
    model.to(device=device, dtype=torch.float64)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, data


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a matched Direct/Indirect checkpoint pair")
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--indirect", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=101)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    direct, direct_data = load_checkpoint(args.direct, "direct", args.device)
    indirect, indirect_data = load_checkpoint(args.indirect, "indirect", args.device)
    if direct_data["problem"] != indirect_data["problem"]:
        raise ValueError(f"checkpoints use different problems: {direct_data['problem']} vs {indirect_data['problem']}")
    problem = make_problem(direct_data["problem"])
    args.output.mkdir(parents=True, exist_ok=True)
    plot_state_comparison(direct, indirect, problem, args.output, args.grid, args.device)
    plot_state_slices(direct, indirect, problem, args.output, args.device)
    plot_controls(direct, indirect, problem, args.output, args.device)
    plot_adjoint(indirect, problem, args.output, args.grid, args.device)
    plot_histories(direct_data, indirect_data, args.output)
    plot_residuals(direct, indirect, problem, args.output, args.grid, args.device)
    plot_metrics(direct, indirect, problem, args.output, args.device)
    print(f"Created 8 figures in {args.output.resolve()}")


if __name__ == "__main__":
    main()
