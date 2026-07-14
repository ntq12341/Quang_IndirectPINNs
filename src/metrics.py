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


def evaluate(model, problem, n_x: int = 101, n_t: int = 101, device: str = "cpu") -> dict[str, float]:
    x_axis = torch.linspace(0, 1, n_x, dtype=torch.float64, device=device)
    t_axis = torch.linspace(0, problem.final_time, n_t, dtype=torch.float64, device=device)
    x, t = torch.meshgrid(x_axis, t_axis, indexing="ij")
    xf, tf = x.reshape(-1, 1), t.reshape(-1, 1)
    result = {"state_relative_l2": relative_l2(model.state(xf, tf), problem.state_exact(xf, tf))}
    errors = []
    for side in (0, 1):
        tt = t_axis.reshape(-1, 1)
        errors.append(relative_l2(model.control(tt, side), problem.control_exact(tt, side)))
    result["control_relative_l2"] = sum(errors) / 2.0
    if hasattr(model, "adjoint"):
        result["adjoint_relative_l2"] = relative_l2(model.adjoint(xf, tf), problem.lambda_exact(xf, tf))
    return result
