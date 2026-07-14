from __future__ import annotations

from dataclasses import dataclass
import torch
from .losses import TrainingBatch, direct_loss, indirect_loss
from .sampling import sample_boundary, sample_initial, sample_interior


@dataclass
class TrainConfig:
    epochs: int = 20000
    n_interior: int = 10000
    n_boundary: int = 2000
    n_initial: int = 500
    learning_rate: float = 1e-3
    resample_every: int = 500
    seed: int = 0
    lbfgs_iterations: int = 0


def make_batch(problem, config: TrainConfig, seed: int, device: str) -> TrainingBatch:
    return TrainingBatch(
        sample_interior(config.n_interior, problem.final_time, device=device, seed=seed),
        sample_initial(config.n_initial, device=device, seed=seed + 1),
        sample_boundary(config.n_boundary, 0, problem.final_time, device=device, seed=seed + 2),
        sample_boundary(config.n_boundary, 1, problem.final_time, device=device, seed=seed + 3),
    )


def train(model, problem, method: str, config: TrainConfig, weights: dict[str, float], *, device: str = "cpu"):
    torch.manual_seed(config.seed)
    model.to(device=device, dtype=torch.float64)
    loss_fn = direct_loss if method == "direct" else indirect_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    batch = make_batch(problem, config, config.seed, device)
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        if epoch and epoch % config.resample_every == 0:
            batch = make_batch(problem, config, config.seed + 4 * epoch, device)
        optimizer.zero_grad(set_to_none=True)
        total, pieces = loss_fn(model, problem, batch, weights)
        total.backward()
        optimizer.step()
        if epoch and epoch % 1000 == 0:
            scheduler.step()
        if epoch == 0 or (epoch + 1) % max(1, min(100, config.epochs)) == 0:
            history.append({"epoch": epoch + 1, "total": float(total.detach()), **{k: float(v.detach()) for k, v in pieces.items()}})
    if config.lbfgs_iterations:
        fixed = make_batch(problem, config, config.seed + 999_999, device)
        lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=config.lbfgs_iterations, line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad(set_to_none=True)
            value, _ = loss_fn(model, problem, fixed, weights)
            value.backward()
            return value
        lbfgs.step(closure)
    return history

