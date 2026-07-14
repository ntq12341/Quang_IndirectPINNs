from __future__ import annotations

import torch


def derivative(value: torch.Tensor, variable: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        value,
        variable,
        grad_outputs=torch.ones_like(value),
        create_graph=True,
        retain_graph=True,
    )[0]


def spatial_derivatives(value: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    first = derivative(value, x)
    return first, derivative(first, x)


def outward_normal_derivative(value: torch.Tensor, x: torch.Tensor, side: int) -> torch.Tensor:
    """Return d_n value at x=0 (side=0) or x=1 (side=1)."""
    if side not in (0, 1):
        raise ValueError("side must be 0 or 1")
    value_x = derivative(value, x)
    return -value_x if side == 0 else value_x

