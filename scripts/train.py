from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.estimators import residual_indicator
from src.metrics import evaluate
from src.models import DirectPINN, IndirectPINN
from src.problems import make_problem
from src.trainers import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PINN for Dirichlet boundary optimal control")
    parser.add_argument("--method", choices=("direct", "indirect"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--problem", choices=("pdf_smoke", "linear_kkt", "nonlinear_kkt"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true", help="Use a tiny CPU-safe end-to-end configuration")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "run.pt")
    args = parser.parse_args()

    config_path = args.config or ROOT / "configs" / f"{args.method}.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    configured_problem = raw.pop("problem")
    problem = make_problem(args.problem or configured_problem)
    weights = raw.pop("weights")
    raw["seed"] = args.seed
    if args.epochs is not None:
        raw["epochs"] = args.epochs
    if args.smoke:
        raw.update(epochs=args.epochs or 5, n_interior=64, n_boundary=32, n_initial=32, lbfgs_iterations=0, resample_every=100)
    cfg = TrainConfig(**raw)
    model = DirectPINN() if args.method == "direct" else IndirectPINN(final_time=problem.final_time)
    history = train(model, problem, args.method, cfg, weights, device=args.device)
    metrics = evaluate(model, problem, device=args.device)
    indicator = residual_indicator(
        model,
        problem,
        n_interior=256 if args.smoke else 20000,
        n_boundary=128 if args.smoke else 4000,
        n_initial=128 if args.smoke else 2000,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"method": args.method, "problem": problem.name, "control_parameterization": "shared_U_psi(x,t)", "state_dict": model.state_dict(), "config": vars(cfg), "weights": weights, "history": history, "metrics": metrics, "indicator": indicator}, args.output)
    print(json.dumps({"metrics": metrics, "indicator": indicator}, indent=2))


if __name__ == "__main__":
    main()
