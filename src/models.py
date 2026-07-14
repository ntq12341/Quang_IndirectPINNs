from __future__ import annotations

import torch
from torch import nn
from .networks import MLP


class BoundaryControl(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (30, 30, 30)):
        super().__init__()
        self.left = MLP(1, hidden)
        self.right = MLP(1, hidden)

    def forward(self, t: torch.Tensor, side: int) -> torch.Tensor:
        return (self.left if side == 0 else self.right)(t)

    def lifting(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - x) * self.left(t) + x * self.right(t)


class DirectPINN(nn.Module):
    def __init__(self, state_hidden: tuple[int, ...] = (50, 50, 50, 50), control_hidden: tuple[int, ...] = (30, 30, 30)):
        super().__init__()
        self.state_net = MLP(2, state_hidden)
        self.control = BoundaryControl(control_hidden)

    def state(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.state_net(torch.cat((x, t), dim=1))


class IndirectPINN(nn.Module):
    def __init__(self, state_hidden: tuple[int, ...] = (50, 50, 50, 50), control_hidden: tuple[int, ...] = (30, 30, 30), adjoint_hidden: tuple[int, ...] = (50, 50, 50, 50), final_time: float = 1.0):
        super().__init__()
        self.state_net = MLP(2, state_hidden)
        self.control = BoundaryControl(control_hidden)
        self.adjoint_net = MLP(2, adjoint_hidden)
        self.final_time = final_time

    def state(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        lifting = self.control.lifting(x, t)
        correction = x * (1.0 - x) * self.state_net(torch.cat((x, t), dim=1))
        return lifting + correction

    def adjoint(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raw = self.adjoint_net(torch.cat((x, t), dim=1))
        return x * (1.0 - x) * (self.final_time - t) * raw

