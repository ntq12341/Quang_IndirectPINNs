# PINNs for Dirichlet Boundary Optimal Control

PyTorch implementation of direct and indirect PINNs for a one-dimensional parabolic PDE with Dirichlet boundary control.

Both methods use one shared control network `U_psi(x,t)`. Its boundary traces are evaluated as `U_psi(0,t)` and `U_psi(1,t)`; the indirect state lifting interpolates these two traces exactly.

The implementation uses

\[
J=\tfrac12\|y-y_d\|_{L^2(Q)}^2+\tfrac\alpha2\|u\|_{L^2(\Sigma)}^2
\]

and the stationarity convention `alpha*u + nu*d_n(lambda) = 0`. The indirect
training loss divides this residual by `alpha` to avoid attenuating control
errors by `alpha^2` after squaring. Verification indicators and residual plots
remain in the original physical scaling. The factor `nu`, scaling, and the
outward-normal signs are covered by tests.

## Quick start

```powershell
python -m pytest -q
python scripts/train.py --method indirect --smoke
python scripts/train.py --method direct --problem pdf_smoke --smoke
python scripts/compare.py outputs/direct-smoke.pt outputs/indirect-smoke.pt
```

Create publication-style figures from a matched checkpoint pair:

```powershell
python scripts/visualize.py --direct outputs/direct-full.pt --indirect outputs/indirect-pdf-full.pt --output outputs/figures/pdf_smoke
python scripts/visualize.py --direct outputs/direct-linear-full.pt --indirect outputs/indirect-full.pt --output outputs/figures/linear_kkt
```

Each command creates state heatmaps and slices, boundary-control curves, the indirect adjoint, Adam loss histories, verification residual maps, stationarity residuals, and a metric comparison at 300 DPI.

Full defaults reproduce the network sizes, point counts, and optimizer budget from the report. Use `--smoke` for a CPU-safe end-to-end check; changing only `--epochs` intentionally leaves the full point counts and L-BFGS budget unchanged.

Available manufactured problems:

- `pdf_smoke`: the zero-control experiment from the PDF.
- `linear_kkt`: a nontrivial manufactured KKT solution with `f=0`.
- `nonlinear_kkt`: the same KKT construction with `f(y)=y^3`.

The nonlinear KKT test has a dedicated indirect configuration that places
more weight on the state and adjoint equations after stationarity
normalization:

```powershell
python scripts/train.py --method indirect --config configs/indirect_nonlinear.json --output outputs/indirect-nonlinear-balanced-full.pt
```

Checkpoints contain model weights, configuration, loss history, metrics, and an independently sampled residual indicator.
