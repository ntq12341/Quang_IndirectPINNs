from __future__ import annotations

import torch
from torch import nn
from .networks import MLP


class BoundaryControl(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (30, 30, 30)):
        super().__init__()
        # One shared network U_psi(x, t), evaluated only at x=0 and x=1.
        self.net = MLP(2, hidden)

    def forward(self, t: torch.Tensor, side: int) -> torch.Tensor:
        if side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        x = torch.full_like(t, float(side))
        return self.net(torch.cat((x, t), dim=1))

    def lifting(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - x) * self.forward(t, 0) + x * self.forward(t, 1)


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
