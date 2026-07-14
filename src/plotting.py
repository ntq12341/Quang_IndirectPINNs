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


def problem_title(problem) -> str:
    return {
        "pdf_smoke": "Cubic reaction–diffusion test",
        "linear_kkt": "Linear boundary-control test",
        "nonlinear_kkt": "Nonlinear boundary-control test",
    }.get(problem.name, problem.name.replace("_", " ").title())


def _array(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
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
    fig = plt.figure(figsize=(15, 9))
    outer = fig.add_gridspec(2, 1, height_ratios=(1, 1), hspace=0.34)
    top = outer[0].subgridspec(1, 3, wspace=0.32)
    bottom = outer[1].subgridspec(1, 2, wspace=0.25)
    top_axes = [fig.add_subplot(top[0, index]) for index in range(3)]
    error_axes = [fig.add_subplot(bottom[0, index]) for index in range(2)]
    extent = (0, 1, 0, problem.final_time)
    for ax, data, title in zip(top_axes, arrays, ("Exact state", "Direct PINN", "Indirect PINN")):
        _heat(ax, data, title, extent, vmin=lo, vmax=hi)
    _heat(error_axes[0], errors[0], f"Direct absolute error (max = {errors[0].max():.3e})", extent, cmap="magma", vmin=0, vmax=err_hi)
    _heat(error_axes[1], errors[1], f"Indirect absolute error (max = {errors[1].max():.3e})", extent, cmap="magma", vmin=0, vmax=err_hi)
    fig.suptitle(f"State comparison — {problem_title(problem)}", fontsize=15, y=0.995)
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
    fig.suptitle(f"State slices — {problem_title(problem)}", fontsize=15)
    _save(fig, out / "state_slices.png")


def plot_controls(direct, indirect, problem, out: Path, device: str):
    t = torch.linspace(0, problem.final_time, 501, dtype=torch.float64, device=device).reshape(-1, 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True)
    for side in (0, 1):
        with torch.no_grad():
            exact = problem.control_exact(t, side)
            dp, ip = direct.control(t, side), indirect.control(t, side)
        axes[side].semilogy(_array(t), np.maximum(np.abs(_array(dp - exact)), 1e-14), "--", lw=2, label="Direct")
        axes[side].semilogy(_array(t), np.maximum(np.abs(_array(ip - exact)), 1e-14), ":", lw=2.5, label="Indirect")
        axes[side].set(title=f"Boundary $x={side}$", xlabel="$t$", ylabel="Absolute control error" if side == 0 else None)
        axes[side].grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle(f"Boundary-control errors — {problem_title(problem)}", fontsize=15)
    _save(fig, out / "control_comparison.png")


def plot_adjoint(indirect, problem, out: Path, n: int, device: str):
    _, _, x, t = make_grid(problem, n, device)
    with torch.no_grad():
        exact = problem.lambda_exact(x, t)
        pred = indirect.adjoint(x.reshape(-1, 1), t.reshape(-1, 1)).reshape_as(x)
    ea, pa = _array(exact), _array(pred)
    error = np.abs(pa - ea)
    signed_limit = max(float(np.max(np.abs(ea))), float(np.max(np.abs(pa))), 1e-14)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    extent = (0, 1, 0, problem.final_time)
    _heat(axes[0], pa, "Indirect adjoint", extent, cmap="coolwarm", vmin=-signed_limit, vmax=signed_limit)
    _heat(axes[1], error, f"Absolute error (max = {error.max():.3e})", extent, cmap="magma")
    fig.suptitle(f"Adjoint variable — {problem_title(problem)}", fontsize=15)
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
    fig.suptitle(f"Verification-grid residuals — {problem_title(problem)}", fontsize=15)
    _save(fig, out / "residual_maps.png")

    t = torch.linspace(0, problem.final_time, 501, dtype=torch.float64, device=device).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for side in (0, 1):
        x = torch.full_like(t, float(side)).requires_grad_(True)
        tt = t.detach().clone().requires_grad_(True)
        residual = stationarity_residual(indirect, problem, PointSet(x, tt), side)
        ax.semilogy(_array(t), np.maximum(np.abs(_array(residual)), 1e-14), label=f"x={side}")
    ax.set(title=f"Stationarity residual — {problem_title(problem)}", xlabel="$t$", ylabel="$|\\alpha u + \\nu\\partial_n\\lambda|$")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, out / "stationarity_residual.png")


def plot_metrics(direct_model, indirect_model, problem, out: Path, device: str = "cpu"):
    labels = ("State error", "Control error")
    x_axis = torch.linspace(0, 1, 101, dtype=torch.float64, device=device)
    t_axis = torch.linspace(0, problem.final_time, 101, dtype=torch.float64, device=device)
    gx, gt = torch.meshgrid(x_axis, t_axis, indexing="ij")
    xf, tf = gx.reshape(-1, 1), gt.reshape(-1, 1)

    def model_rmse(model):
        with torch.no_grad():
            state_error = model.state(xf, tf) - problem.state_exact(xf, tf)
            state_value = float(torch.sqrt(torch.mean(state_error.square())))
            boundary_mse = []
            times = t_axis.reshape(-1, 1)
            for side in (0, 1):
                error = model.control(times, side) - problem.control_exact(times, side)
                boundary_mse.append(torch.mean(error.square()))
            control_value = float(torch.sqrt(sum(boundary_mse) / 2.0))
        return [state_value, control_value]

    direct = model_rmse(direct_model)
    indirect = model_rmse(indirect_model)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    direct_bars = ax.bar(x - width / 2, direct, width, label="Direct")
    indirect_bars = ax.bar(x + width / 2, indirect, width, label="Indirect")
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    titles = {
        "linear_kkt": "RMSE comparison — Linear boundary-control test",
        "pdf_smoke": "RMSE comparison — Cubic reaction–diffusion test",
        "nonlinear_kkt": "RMSE comparison — Nonlinear boundary-control test",
    }
    ax.set_ylabel("Root mean squared error (RMSE)")
    ax.set_title(titles.get(problem.name, f"RMSE comparison — {problem.name}"))
    for bars in (direct_bars, indirect_bars):
        ax.bar_label(bars, labels=[f"{bar.get_height():.2e}" for bar in bars], padding=4, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, out / "metric_comparison.png")
