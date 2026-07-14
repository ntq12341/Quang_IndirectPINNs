from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int = 1):
        super().__init__()
        widths = (in_dim, *hidden, out_dim)
        layers: list[nn.Module] = []
        for left, right in zip(widths[:-2], widths[1:-1]):
            linear = nn.Linear(left, right)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.Tanh()))
        final = nn.Linear(widths[-2], widths[-1])
        nn.init.xavier_uniform_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

