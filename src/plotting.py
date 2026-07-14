from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

from .residuals import adjoint_residual, state_residual, stationarity_residual
from .sampling import PointSet


def _array(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _heat(ax, data, title: str, extent, *, cmap="viridis", log=False, vmin=None, vmax=None):
    if log:
        positive = np.maximum(data, 1e-14)
        image = ax.imshow(positive.T, origin="lower", aspect="auto", extent=extent, cmap=cmap, norm=LogNorm(vmin=max(vmin or positive.min(), 1e-14), vmax=vmax or positive.max()))
    else:
        image = ax.imshow(data.T, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set(title=title, xlabel="$x$", ylabel="$t$")
    plt.colorbar(image, ax=ax, shrink=0.85)


def make_grid(problem, n: int, device: str):
    xa = torch.linspace(0, 1, n, dtype=torch.float64, device=device)
    ta = torch.linspace(0, problem.final_time, n, dtype=torch.float64, device=device)
    x, t = torch.meshgrid(xa, ta, indexing="ij")
    return xa, ta, x, t


def predict_grid(model, x, t):
    with torch.no_grad():
        return model.state(x.reshape(-1, 1), t.reshape(-1, 1)).reshape_as(x)


def plot_state_comparison(direct, indirect, problem, out: Path, n: int, device: str):
    _, _, x, t = make_grid(problem, n, device)
    exact = problem.state_exact(x, t)
    dp, ip = predict_grid(direct, x, t), predict_grid(indirect, x, t)
    arrays = [_array(v) for v in (exact, dp, ip)]
    errors = [np.abs(arrays[1] - arrays[0]), np.abs(arrays[2] - arrays[0])]
    lo, hi = min(a.min() for a in arrays), max(a.max() for a in arrays)
    err_hi = max(e.max() for e in errors)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    extent = (0, 1, 0, problem.final_time)
    for ax, data, title in zip(axes[0], arrays, ("Exact state", "Direct PINN", "Indirect PINN")):
        _heat(ax, data, title, extent, vmin=lo, vmax=hi)
    _heat(axes[1, 0], errors[0], "Direct absolute error", extent, cmap="magma", vmin=0, vmax=err_hi)
    _heat(axes[1, 1], errors[1], "Indirect absolute error", extent, cmap="magma", vmin=0, vmax=err_hi)
    axes[1, 2].axis("off")
    axes[1, 2].text(0.05, 0.7, f"Direct max error: {errors[0].max():.3e}\nIndirect max error: {errors[1].max():.3e}\n\nShared color scales are used.", fontsize=12, va="top")
    fig.suptitle(f"State comparison — {problem.name}", fontsize=15)
    _save(fig, out / "state_comparison.png")


def plot_state_slices(direct, indirect, problem, out: Path, device: str):
    x = torch.linspace(0, 1, 401, dtype=torch.float64, device=device).reshape(-1, 1)
    times = (0.25, 0.5, 0.75, 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, time in zip(axes.flat, times):
        t = torch.full_like(x, time)
        with torch.no_grad():
            exact = problem.state_exact(x, t)
            dp, ip = direct.state(x, t), indirect.state(x, t)
        ax.plot(_array(x), _array(exact), "k-", lw=2, label="Exact")
        ax.plot(_array(x), _array(dp), "--", label="Direct")
        ax.plot(_array(x), _array(ip), ":", lw=2, label="Indirect")
        ax.set(title=f"$t={time:g}$", xlabel="$x$", ylabel="$y$")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle(f"State slices — {problem.name}", fontsize=15)
    _save(fig, out / "state_slices.png")


def plot_controls(direct, indirect, problem, out: Path, device: str):
    t = torch.linspace(0, problem.final_time, 501, dtype=torch.float64, device=device).reshape(-1, 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for side in (0, 1):
        with torch.no_grad():
            exact = problem.control_exact(t, side)
            dp, ip = direct.control(t, side), indirect.control(t, side)
        axes[0, side].plot(_array(t), _array(exact), "k-", lw=2, label="Exact")
        axes[0, side].plot(_array(t), _array(dp), "--", label="Direct")
        axes[0, side].plot(_array(t), _array(ip), ":", lw=2, label="Indirect")
        axes[0, side].set(title=f"Control at x={side}", ylabel="$u(t)$")
        axes[1, side].semilogy(_array(t), np.maximum(np.abs(_array(dp - exact)), 1e-14), "--", label="Direct error")
        axes[1, side].semilogy(_array(t), np.maximum(np.abs(_array(ip - exact)), 1e-14), ":", lw=2, label="Indirect error")
        axes[1, side].set(xlabel="$t$", ylabel="absolute error")
        for ax in axes[:, side]: ax.grid(alpha=0.25)
    axes[0, 0].legend()
    axes[1, 0].legend()
    fig.suptitle(f"Boundary controls — {problem.name}", fontsize=15)
    _save(fig, out / "control_comparison.png")


def plot_adjoint(indirect, problem, out: Path, n: int, device: str):
    _, _, x, t = make_grid(problem, n, device)
    with torch.no_grad():
        exact = problem.lambda_exact(x, t)
        pred = indirect.adjoint(x.reshape(-1, 1), t.reshape(-1, 1)).reshape_as(x)
    ea, pa = _array(exact), _array(pred)
    error = np.abs(pa - ea)
    lo, hi = min(ea.min(), pa.min()), max(ea.max(), pa.max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    extent = (0, 1, 0, problem.final_time)
    _heat(axes[0], ea, "Exact adjoint", extent, vmin=lo, vmax=hi)
    _heat(axes[1], pa, "Indirect adjoint", extent, vmin=lo, vmax=hi)
    _heat(axes[2], error, "Absolute error", extent, cmap="magma")
    fig.suptitle(f"Adjoint variable — {problem.name}", fontsize=15)
    _save(fig, out / "indirect_adjoint.png")


def plot_histories(direct_data, indirect_data, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title in ((axes[0], direct_data, "Direct PINN"), (axes[1], indirect_data, "Indirect PINN")):
        history = data.get("history", [])
        if history:
            keys = [key for key in history[0] if key != "epoch"]
            epochs = [row["epoch"] for row in history]
            for key in keys:
                ax.semilogy(epochs, [max(row[key], 1e-16) for row in history], label=key)
        ax.set(title=title, xlabel="Adam epoch", ylabel="loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Training histories (Adam stage; L-BFGS history was not stored)", fontsize=14)
    _save(fig, out / "loss_history.png")


def _residual_grid(model, problem, kind: str, n: int, device: str, chunk: int = 256):
    _, _, gx, gt = make_grid(problem, n, device)
    values = []
    flat_x, flat_t = gx.reshape(-1, 1), gt.reshape(-1, 1)
    for start in range(0, flat_x.shape[0], chunk):
        x = flat_x[start:start + chunk].detach().clone().requires_grad_(True)
        t = flat_t[start:start + chunk].detach().clone().requires_grad_(True)
        points = PointSet(x, t)
        residual = state_residual(model, problem, points) if kind == "state" else adjoint_residual(model, problem, points)
        values.append(residual.detach().cpu())
        del residual, points, x, t
    return torch.cat(values).reshape(n, n)


def plot_residuals(direct, indirect, problem, out: Path, n: int, device: str):
    direct_state = _array(_residual_grid(direct, problem, "state", n, device)).__abs__()
    indirect_state = _array(_residual_grid(indirect, problem, "state", n, device)).__abs__()
    indirect_adjoint = _array(_residual_grid(indirect, problem, "adjoint", n, device)).__abs__()
    all_positive = np.concatenate([a.ravel() for a in (direct_state, indirect_state, indirect_adjoint)])
    vmin, vmax = max(np.percentile(all_positive, 1), 1e-12), max(all_positive.max(), 1e-11)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    extent = (0, 1, 0, problem.final_time)
    for ax, data, title in zip(axes, (direct_state, indirect_state, indirect_adjoint), ("Direct $|R_y|$", "Indirect $|R_y|$", "Indirect $|R_\\lambda|$")):
        _heat(ax, data, title, extent, cmap="magma", log=True, vmin=vmin, vmax=vmax)
    fig.suptitle(f"Verification-grid residuals — {problem.name}", fontsize=15)
    _save(fig, out / "residual_maps.png")

    t = torch.linspace(0, problem.final_time, 501, dtype=torch.float64, device=device).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for side in (0, 1):
        x = torch.full_like(t, float(side)).requires_grad_(True)
        tt = t.detach().clone().requires_grad_(True)
        residual = stationarity_residual(indirect, problem, PointSet(x, tt), side)
        ax.semilogy(_array(t), np.maximum(np.abs(_array(residual)), 1e-14), label=f"x={side}")
    ax.set(title=f"Stationarity residual — {problem.name}", xlabel="$t$", ylabel="$|\\alpha u + \\nu\\partial_n\\lambda|$")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, out / "stationarity_residual.png")


def plot_metrics(direct_data, indirect_data, out: Path):
    keys = ("state_relative_l2", "control_relative_l2")
    labels = ("State error", "Control error")
    direct = [direct_data["metrics"][key] for key in keys]
    indirect = [indirect_data["metrics"][key] for key in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    ax.bar(x - width / 2, direct, width, label="Direct")
    ax.bar(x + width / 2, indirect, width, label="Indirect")
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_ylabel("reported error")
    ax.set_title("Accuracy comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, out / "metric_comparison.png")
