from __future__ import annotations

import math
import torch
from .residuals import adjoint_residual, state_residual, stationarity_residual
from .sampling import sample_boundary, sample_initial, sample_interior


def residual_indicator(model, problem, n_interior: int = 20000, n_boundary: int = 4000, n_initial: int = 2000, seed: int = 123456, device: str = "cpu") -> dict[str, float]:
    interior = sample_interior(n_interior, problem.final_time, device=device, seed=seed)
    initial = sample_initial(n_initial, device=device, seed=seed + 1)
    left = sample_boundary(n_boundary, 0, problem.final_time, device=device, seed=seed + 2)
    right = sample_boundary(n_boundary, 1, problem.final_time, device=device, seed=seed + 3)
    terms = {
        "state": float(torch.mean(state_residual(model, problem, interior).square()).detach()),
        "initial": float(torch.mean((model.state(initial.x, initial.t) - problem.initial(initial.x)).square()).detach()),
    }
    if hasattr(model, "adjoint"):
        terms["adjoint"] = float(torch.mean(adjoint_residual(model, problem, interior).square()).detach())
        # L2(Sigma) uses counting measure over the two endpoints in 1D.
        terms["stationarity"] = sum(float(torch.mean(stationarity_residual(model, problem, points, side).square()).detach()) for side, points in ((0, left), (1, right)))
    terms["eta"] = math.sqrt(sum(terms.values()))
    return terms
