from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class PointSet:
    x: torch.Tensor
    t: torch.Tensor


def _leaf(values: torch.Tensor) -> torch.Tensor:
    return values.detach().clone().requires_grad_(True)


def sample_interior(n: int, final_time: float, *, device: str = "cpu", seed: int | None = None) -> PointSet:
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    x = torch.rand((n, 1), generator=generator, device=device, dtype=torch.float64)
    t = final_time * torch.rand((n, 1), generator=generator, device=device, dtype=torch.float64)
    return PointSet(_leaf(x), _leaf(t))


def sample_initial(n: int, *, device: str = "cpu", seed: int | None = None) -> PointSet:
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    x = torch.rand((n, 1), generator=generator, device=device, dtype=torch.float64)
    return PointSet(_leaf(x), _leaf(torch.zeros_like(x)))


def sample_boundary(n: int, side: int, final_time: float, *, device: str = "cpu", seed: int | None = None) -> PointSet:
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    t = final_time * torch.rand((n, 1), generator=generator, device=device, dtype=torch.float64)
    x = torch.full_like(t, float(side))
    return PointSet(_leaf(x), _leaf(t))

