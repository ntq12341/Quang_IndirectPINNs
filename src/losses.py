from __future__ import annotations

from dataclasses import dataclass
import torch
from .residuals import state_residual, adjoint_residual, stationarity_residual
from .sampling import PointSet


def mse(value: torch.Tensor) -> torch.Tensor:
    return torch.mean(value.square())


@dataclass
class TrainingBatch:
    interior: PointSet
    initial: PointSet
    left: PointSet
    right: PointSet


def direct_loss(model, problem, batch: TrainingBatch, weights: dict[str, float]):
    y0 = model.state(batch.initial.x, batch.initial.t)
    pieces = {
        "state": mse(state_residual(model, problem, batch.interior)),
        "initial": mse(y0 - problem.initial(batch.initial.x)),
    }
    boundary_errors = []
    control_costs = []
    for side, points in ((0, batch.left), (1, batch.right)):
        u = model.control(points.t, side)
        boundary_errors.append(mse(model.state(points.x, points.t) - u))
        control_costs.append(mse(u))
    pieces["boundary"] = sum(boundary_errors) / 2.0
    pieces["tracking"] = 0.5 * mse(model.state(batch.interior.x, batch.interior.t) - problem.desired(batch.interior.x, batch.interior.t))
    # Surface measure in 1D is counting measure on {0, 1}: sum both sides.
    pieces["control"] = 0.5 * problem.alpha * sum(control_costs)
    total = (
        weights["state"] * pieces["state"]
        + weights["boundary"] * pieces["boundary"]
        + weights["initial"] * pieces["initial"]
        + pieces["tracking"]
        + pieces["control"]
    )
    return total, pieces


def indirect_loss(model, problem, batch: TrainingBatch, weights: dict[str, float]):
    y0 = model.state(batch.initial.x, batch.initial.t)
    pieces = {
        "state": mse(state_residual(model, problem, batch.interior)),
        "adjoint": mse(adjoint_residual(model, problem, batch.interior)),
        "stationarity": 0.5 * (
            mse(stationarity_residual(model, problem, batch.left, 0))
            + mse(stationarity_residual(model, problem, batch.right, 1))
        ),
        "initial": mse(y0 - problem.initial(batch.initial.x)),
    }
    total = sum(weights[name] * value for name, value in pieces.items())
    return total, pieces
