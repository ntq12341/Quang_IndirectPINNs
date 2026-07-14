from __future__ import annotations

import torch
from .autodiff import derivative, spatial_derivatives, outward_normal_derivative
from .problems import ManufacturedProblem
from .sampling import PointSet


def state_residual(model, problem: ManufacturedProblem, points: PointSet) -> torch.Tensor:
    y = model.state(points.x, points.t)
    y_t = derivative(y, points.t)
    _, y_xx = spatial_derivatives(y, points.x)
    return y_t - problem.nu * y_xx + problem.f(y) - problem.source(points.x, points.t)


def adjoint_residual(model, problem: ManufacturedProblem, points: PointSet) -> torch.Tensor:
    y = model.state(points.x, points.t)
    lam = model.adjoint(points.x, points.t)
    lam_t = derivative(lam, points.t)
    _, lam_xx = spatial_derivatives(lam, points.x)
    return -lam_t - problem.nu * lam_xx + problem.f_prime(y) * lam + y - problem.desired(points.x, points.t)


def stationarity_residual(model, problem: ManufacturedProblem, points: PointSet, side: int) -> torch.Tensor:
    lam = model.adjoint(points.x, points.t)
    normal = outward_normal_derivative(lam, points.x, side)
    control = model.control(points.t, side)
    return problem.alpha * control + problem.nu * normal

