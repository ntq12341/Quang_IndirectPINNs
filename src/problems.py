from __future__ import annotations

from dataclasses import dataclass
import math
import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class ManufacturedProblem:
    name: str
    nu: float = 0.1
    alpha: float = 0.01
    final_time: float = 1.0
    amplitude: float = 0.05

    def f(self, y: Tensor) -> Tensor:
        return torch.zeros_like(y) if self.name == "linear_kkt" else y**3

    def f_prime(self, y: Tensor) -> Tensor:
        return torch.zeros_like(y) if self.name == "linear_kkt" else 3.0 * y**2

    def lambda_exact(self, x: Tensor, t: Tensor) -> Tensor:
        if self.name == "pdf_smoke":
            return torch.zeros_like(x)
        return self.amplitude * x * (1.0 - x) * torch.sin(math.pi * t)

    def lambda_t(self, x: Tensor, t: Tensor) -> Tensor:
        if self.name == "pdf_smoke":
            return torch.zeros_like(x)
        return self.amplitude * math.pi * x * (1.0 - x) * torch.cos(math.pi * t)

    def lambda_xx(self, x: Tensor, t: Tensor) -> Tensor:
        if self.name == "pdf_smoke":
            return torch.zeros_like(x)
        return -2.0 * self.amplitude * torch.sin(math.pi * t)

    def control_exact(self, t: Tensor, side: int) -> Tensor:
        if self.name == "pdf_smoke":
            return torch.zeros_like(t)
        # Both outward derivatives equal -A sin(pi t).
        return (self.nu * self.amplitude / self.alpha) * torch.sin(math.pi * t)

    def state_exact(self, x: Tensor, t: Tensor) -> Tensor:
        if self.name == "pdf_smoke":
            return torch.sin(math.pi * x) * torch.exp(-t)
        boundary = self.control_exact(t, 0)
        return boundary + torch.sin(math.pi * x) * torch.exp(-t)

    def initial(self, x: Tensor) -> Tensor:
        return self.state_exact(x, torch.zeros_like(x))

    def source(self, x: Tensor, t: Tensor) -> Tensor:
        # Exact derivatives for y = b(t) + sin(pi*x) exp(-t).
        interior = torch.sin(math.pi * x) * torch.exp(-t)
        if self.name == "pdf_smoke":
            y_t = -interior
        else:
            y_t = (
                self.nu * self.amplitude * math.pi / self.alpha * torch.cos(math.pi * t)
                - interior
            )
        y_xx = -(math.pi**2) * interior
        y = self.state_exact(x, t)
        return y_t - self.nu * y_xx + self.f(y)

    def desired(self, x: Tensor, t: Tensor) -> Tensor:
        y = self.state_exact(x, t)
        lam = self.lambda_exact(x, t)
        # R_lambda = -lam_t - nu*lam_xx + f'(y)*lam + y - yd.
        return y - self.lambda_t(x, t) - self.nu * self.lambda_xx(x, t) + self.f_prime(y) * lam


def make_problem(name: str = "linear_kkt", **kwargs: float) -> ManufacturedProblem:
    if name not in {"linear_kkt", "nonlinear_kkt", "pdf_smoke"}:
        raise ValueError(f"unknown problem: {name}")
    return ManufacturedProblem(name=name, **kwargs)

