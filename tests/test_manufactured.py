import pytest
import torch
from src.autodiff import derivative, outward_normal_derivative, spatial_derivatives
from src.problems import make_problem


@pytest.mark.parametrize("name", ["pdf_smoke", "linear_kkt", "nonlinear_kkt"])
def test_manufactured_solution_satisfies_state_equation(name):
    problem = make_problem(name)
    x = torch.linspace(0.05, 0.95, 19, dtype=torch.float64).reshape(-1, 1).requires_grad_()
    t = torch.linspace(0.03, 0.97, 19, dtype=torch.float64).reshape(-1, 1).requires_grad_()
    y = problem.state_exact(x, t)
    _, y_xx = spatial_derivatives(y, x)
    residual = derivative(y, t) - problem.nu * y_xx + problem.f(y) - problem.source(x, t)
    assert torch.max(torch.abs(residual)) < 1e-10


@pytest.mark.parametrize("name", ["linear_kkt", "nonlinear_kkt"])
def test_manufactured_solution_satisfies_adjoint_and_stationarity(name):
    problem = make_problem(name)
    x = torch.linspace(0.05, 0.95, 19, dtype=torch.float64).reshape(-1, 1).requires_grad_()
    t = torch.linspace(0.03, 0.97, 19, dtype=torch.float64).reshape(-1, 1).requires_grad_()
    y = problem.state_exact(x, t)
    lam = problem.lambda_exact(x, t)
    _, lam_xx = spatial_derivatives(lam, x)
    residual = -derivative(lam, t) - problem.nu * lam_xx + problem.f_prime(y) * lam + y - problem.desired(x, t)
    assert torch.max(torch.abs(residual)) < 1e-10
    for side in (0, 1):
        xb = torch.full((19, 1), float(side), dtype=torch.float64, requires_grad=True)
        tb = t.detach().clone().requires_grad_()
        lamb = problem.lambda_exact(xb, tb)
        stationarity = problem.alpha * problem.control_exact(tb, side) + problem.nu * outward_normal_derivative(lamb, xb, side)
        assert torch.max(torch.abs(stationarity)) < 1e-10

