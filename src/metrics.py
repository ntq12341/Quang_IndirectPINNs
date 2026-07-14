from __future__ import annotations

import torch


def relative_l2(predicted: torch.Tensor, exact: torch.Tensor) -> float:
    predicted = predicted.detach()
    exact = exact.detach()
    denominator = torch.linalg.vector_norm(exact)
    numerator = torch.linalg.vector_norm(predicted - exact)
    if float(denominator) <= torch.finfo(exact.dtype).eps:
        return float(torch.sqrt(torch.mean((predicted - exact).square())))
    return float(numerator / denominator)


def rmse(predicted: torch.Tensor, exact: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((predicted.detach() - exact.detach()).square())))


def evaluate(model, problem, n_x: int = 101, n_t: int = 101, device: str = "cpu") -> dict[str, float]:
    x_axis = torch.linspace(0, 1, n_x, dtype=torch.float64, device=device)
    t_axis = torch.linspace(0, problem.final_time, n_t, dtype=torch.float64, device=device)
    x, t = torch.meshgrid(x_axis, t_axis, indexing="ij")
    xf, tf = x.reshape(-1, 1), t.reshape(-1, 1)
    state_predicted = model.state(xf, tf)
    state_exact = problem.state_exact(xf, tf)
    result = {
        "state_relative_l2": relative_l2(state_predicted, state_exact),
        "state_rmse": rmse(state_predicted, state_exact),
    }
    relative_errors = []
    squared_errors = []
    for side in (0, 1):
        tt = t_axis.reshape(-1, 1)
        predicted = model.control(tt, side)
        exact = problem.control_exact(tt, side)
        relative_errors.append(relative_l2(predicted, exact))
        squared_errors.append(torch.mean((predicted.detach() - exact.detach()).square()))
    result["control_relative_l2"] = sum(relative_errors) / 2.0
    result["control_rmse"] = float(torch.sqrt(sum(squared_errors) / 2.0))
    if hasattr(model, "adjoint"):
        result["adjoint_relative_l2"] = relative_l2(model.adjoint(xf, tf), problem.lambda_exact(xf, tf))
    return result
