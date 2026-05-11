"""PINN solver for the P2D solid-phase spherical diffusion equation.

The script solves the dimensionless form

    dc/dtau = d2c/dx2 + 2/x * dc/dx,      0 < x < 1, 0 < tau <= tau_max

with

    c(x, 0) = 0
    dc/dx(0, tau) = 0
    dc/dx(1, tau) = -phi(tau)

Here x = r / R_s and tau = D_s t / R_s^2.  The unknown c is a scaled
concentration perturbation around the initial solid concentration.  For a
physical concentration c_s, use

    c_s(r, t) = c_s0 + c_scale * c(r / R_s, D_s t / R_s^2)

and set phi = J_surf R_s / (D_s c_scale), where the surface flux boundary is

    -D_s * dc_s/dr |_{r=R_s} = J_surf.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp


torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class Config:
    tau_max: float = 0.4
    phi: float = 0.2
    hidden_width: int = 64
    hidden_layers: int = 4
    n_collocation: int = 5000
    n_initial: int = 512
    n_boundary: int = 900
    adam_steps: int = 3000
    lbfgs_steps: int = 500
    learning_rate: float = 1e-3
    seed: int = 2026
    output_dir: Path = Path("results")


class MLP(torch.nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        modules: list[torch.nn.Module] = [torch.nn.Linear(2, width), torch.nn.Tanh()]
        for _ in range(layers - 1):
            modules.extend([torch.nn.Linear(width, width), torch.nn.Tanh()])
        modules.append(torch.nn.Linear(width, 1))
        self.net = torch.nn.Sequential(*modules)

        for module in self.net:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x_tau: torch.Tensor) -> torch.Tensor:
        return self.net(x_tau)


class HardConstrainedPINN(torch.nn.Module):
    """Network whose ansatz exactly satisfies IC and the spherical-center BC."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.tau_max = config.tau_max
        self.core = MLP(width=config.hidden_width, layers=config.hidden_layers)

    def forward(self, x_tau: torch.Tensor) -> torch.Tensor:
        x = x_tau[:, 0:1]
        tau = x_tau[:, 1:2]
        z = 2.0 * x**2 - 1.0
        s = 2.0 * tau / self.tau_max - 1.0
        raw = self.core(torch.cat([z, s], dim=1))
        return tau * raw


def make_model(config: Config) -> HardConstrainedPINN:
    return HardConstrainedPINN(config)


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def phi_of_tau(tau: torch.Tensor, phi: float) -> torch.Tensor:
    """Dimensionless molar flux. Replace this with a current profile if needed."""
    return torch.full_like(tau, phi)


def sample_training_points(config: Config, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(config.seed)

    # Avoid x = 0 in the PDE residual because the spherical operator has 2/x.
    eps = 1e-4
    n_random = int(0.7 * config.n_collocation)
    n_early = config.n_collocation - n_random
    x_f_random = eps + (1.0 - eps) * torch.rand(n_random, 1, generator=generator, device=device)
    tau_f_random = config.tau_max * torch.rand(n_random, 1, generator=generator, device=device)
    x_f_early = eps + (1.0 - eps) * torch.rand(n_early, 1, generator=generator, device=device)
    tau_f_early = config.tau_max * torch.rand(n_early, 1, generator=generator, device=device) ** 2
    x_f = torch.cat([x_f_random, x_f_early], dim=0)
    tau_f = torch.cat([tau_f_random, tau_f_early], dim=0)

    x_i = torch.rand(config.n_initial, 1, generator=generator, device=device)
    tau_i = torch.zeros_like(x_i)

    tau_grid = torch.linspace(1e-6, config.tau_max, config.n_boundary, device=device).reshape(-1, 1)
    tau_random = config.tau_max * torch.rand(config.n_boundary, 1, generator=generator, device=device) ** 2
    tau_b = torch.cat([tau_grid, tau_random], dim=0)
    x_center = torch.zeros_like(tau_b)
    x_surface = torch.ones_like(tau_b)

    return {
        "x_f": x_f,
        "tau_f": tau_f,
        "x_i": x_i,
        "tau_i": tau_i,
        "x_center": x_center,
        "x_surface": x_surface,
        "tau_b": tau_b,
    }


def loss_terms(model: HardConstrainedPINN, pts: dict[str, torch.Tensor], config: Config) -> dict[str, torch.Tensor]:
    x_f = pts["x_f"].detach().clone().requires_grad_(True)
    tau_f = pts["tau_f"].detach().clone().requires_grad_(True)
    c_f = model(torch.cat([x_f, tau_f], dim=1))
    c_x = grad(c_f, x_f)
    c_tau = grad(c_f, tau_f)
    c_xx = grad(c_x, x_f)
    residual = c_tau - c_xx - 2.0 * c_x / x_f
    pde_loss = torch.mean(residual**2)

    x_surface = pts["x_surface"].detach().clone().requires_grad_(True)
    tau_surface = pts["tau_b"].detach().clone().requires_grad_(True)
    c_surface = model(torch.cat([x_surface, tau_surface], dim=1))
    surface_slope = grad(c_surface, x_surface)
    flux_loss = torch.mean((surface_slope + phi_of_tau(tau_surface, config.phi)) ** 2)

    c_i = model(torch.cat([pts["x_i"], pts["tau_i"]], dim=1))
    initial_loss = torch.mean(c_i**2)
    x_center = pts["x_center"].detach().clone().requires_grad_(True)
    tau_center = pts["tau_b"].detach().clone().requires_grad_(True)
    c_center = model(torch.cat([x_center, tau_center], dim=1))
    center_slope = grad(c_center, x_center)
    center_loss = torch.mean(center_slope**2)

    return {
        "pde": pde_loss,
        "initial": initial_loss,
        "center": center_loss,
        "flux": flux_loss,
    }


def total_loss(model: HardConstrainedPINN, pts: dict[str, torch.Tensor], config: Config) -> tuple[torch.Tensor, dict[str, float]]:
    terms = loss_terms(model, pts, config)
    weights = {"pde": 1.0, "initial": 0.0, "center": 0.0, "flux": 40.0}
    loss = sum(weights[name] * value for name, value in terms.items())
    return loss, {name: float(value.detach().cpu()) for name, value in terms.items()}


def train(model: HardConstrainedPINN, pts: dict[str, torch.Tensor], config: Config) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(config.adam_steps * 0.45)), max(1, int(config.adam_steps * 0.75))],
        gamma=0.35,
    )

    for step in range(1, config.adam_steps + 1):
        optimizer.zero_grad()
        loss, terms = total_loss(model, pts, config)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0:
            row = {"step": float(step), "loss": float(loss.detach().cpu()), **terms}
            history.append(row)
            print(
                f"Adam {step:5d} | loss={row['loss']:.3e} "
                f"pde={row['pde']:.3e} ic={row['initial']:.3e} "
                f"center={row['center']:.3e} flux={row['flux']:.3e}"
            )

    if config.lbfgs_steps <= 0:
        return history

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=config.lbfgs_steps,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        loss, _ = total_loss(model, pts, config)
        loss.backward()
        return loss

    lbfgs.step(closure)
    loss, terms = total_loss(model, pts, config)
    row = {"step": float(config.adam_steps + config.lbfgs_steps), "loss": float(loss.detach().cpu()), **terms}
    history.append(row)
    print(
        f"LBFGS done | loss={row['loss']:.3e} pde={row['pde']:.3e} "
        f"ic={row['initial']:.3e} center={row['center']:.3e} flux={row['flux']:.3e}"
    )

    return history


def finite_volume_reference(config: Config, nx: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference solution on x in [0, 1] for sanity checking the PINN result."""
    edges = np.linspace(0.0, 1.0, nx + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    volumes = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
    areas = edges**2
    dx = edges[1] - edges[0]

    def rhs(_tau: float, c: np.ndarray) -> np.ndarray:
        flux = np.zeros(nx + 1)
        dc = np.diff(c) / dx
        flux[1:nx] = -areas[1:nx] * dc
        flux[0] = 0.0
        flux[nx] = areas[nx] * config.phi
        return -(flux[1:] - flux[:-1]) / volumes

    t_eval = np.linspace(0.0, config.tau_max, 81)
    sol = solve_ivp(rhs, (0.0, config.tau_max), np.zeros(nx), t_eval=t_eval, method="BDF")
    if not sol.success:
        raise RuntimeError(sol.message)
    return centers, sol.t, sol.y.T


@torch.no_grad()
def predict_grid(model: HardConstrainedPINN, config: Config, device: torch.device, nx: int = 241, nt: int = 121) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nx)
    tau = np.linspace(0.0, config.tau_max, nt)
    xx, tt = np.meshgrid(x, tau)
    inputs = torch.tensor(np.column_stack([xx.ravel(), tt.ravel()]), device=device)
    c = model(inputs).cpu().numpy().reshape(nt, nx)
    return x, tau, c


def evaluate_autograd(
    model: HardConstrainedPINN,
    config: Config,
    device: torch.device,
    nx: int = 121,
    nt: int = 121,
) -> dict[str, float | np.ndarray]:
    x = np.linspace(1e-4, 1.0, nx)
    tau = np.linspace(1e-6, config.tau_max, nt)
    xx, tt = np.meshgrid(x, tau)
    x_t = torch.tensor(xx.ravel()[:, None], device=device, requires_grad=True)
    tau_t = torch.tensor(tt.ravel()[:, None], device=device, requires_grad=True)
    c = model(torch.cat([x_t, tau_t], dim=1))
    c_x = grad(c, x_t)
    c_tau = grad(c, tau_t)
    c_xx = grad(c_x, x_t)
    residual = (c_tau - c_xx - 2.0 * c_x / x_t).detach().cpu().numpy().reshape(nt, nx)
    post_mask = tau >= 1e-4

    tau_b = torch.tensor(tau[:, None], device=device, requires_grad=True)
    x_surface = torch.ones_like(tau_b, requires_grad=True)
    c_surface = model(torch.cat([x_surface, tau_b], dim=1))
    surface_slope = grad(c_surface, x_surface).detach().cpu().numpy().ravel()

    x_center = torch.zeros_like(tau_b, requires_grad=True)
    c_center = model(torch.cat([x_center, tau_b], dim=1))
    center_slope = grad(c_center, x_center).detach().cpu().numpy().ravel()

    x_ic = torch.linspace(0.0, 1.0, nx, device=device).reshape(-1, 1)
    tau_ic = torch.zeros_like(x_ic)
    c_ic = model(torch.cat([x_ic, tau_ic], dim=1)).detach().cpu().numpy().ravel()

    return {
        "residual_grid": residual,
        "residual_x": x,
        "residual_tau": tau,
        "pde_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "pde_residual_max_abs": float(np.max(np.abs(residual))),
        "pde_residual_rms_tau_ge_1e_4": float(np.sqrt(np.mean(residual[post_mask] ** 2))),
        "pde_residual_p99_abs_tau_ge_1e_4": float(np.percentile(np.abs(residual[post_mask]), 99.0)),
        "pde_residual_max_abs_tau_ge_1e_4": float(np.max(np.abs(residual[post_mask]))),
        "surface_flux_max_abs_error": float(np.max(np.abs(surface_slope + config.phi))),
        "surface_flux_rms_error": float(np.sqrt(np.mean((surface_slope + config.phi) ** 2))),
        "surface_flux_rms_error_tau_ge_1e_4": float(np.sqrt(np.mean((surface_slope[post_mask] + config.phi) ** 2))),
        "surface_flux_max_abs_error_tau_ge_1e_4": float(np.max(np.abs(surface_slope[post_mask] + config.phi))),
        "center_slope_max_abs": float(np.max(np.abs(center_slope))),
        "initial_max_abs": float(np.max(np.abs(c_ic))),
        "initial_rms": float(np.sqrt(np.mean(c_ic**2))),
    }


def save_outputs(model: HardConstrainedPINN, history: list[dict[str, float]], config: Config, device: torch.device) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    x, tau, c = predict_grid(model, config, device)
    x_ref, tau_ref, c_ref = finite_volume_reference(config)

    ref_interp = np.vstack([np.interp(tau, tau_ref, c_ref[:, j]) for j in range(c_ref.shape[1])]).T
    ref_on_pinn_grid = np.vstack([np.interp(x, x_ref, row) for row in ref_interp])
    error = c - ref_on_pinn_grid
    rmse = float(np.sqrt(np.mean(error**2)))
    max_error = float(np.max(np.abs(error)))

    volume_average = np.array([3.0 * np.trapezoid(x * x * row, x) for row in c])
    expected_average = -3.0 * config.phi * tau
    mass_error = volume_average - expected_average
    diagnostics = evaluate_autograd(model, config, device)

    np.savez(
        config.output_dir / "solid_diffusion_pinn_solution.npz",
        x=x,
        tau=tau,
        c=c,
        x_ref=x_ref,
        tau_ref=tau_ref,
        c_ref=c_ref,
        ref_on_pinn_grid=ref_on_pinn_grid,
        error=error,
        volume_average=volume_average,
        expected_average=expected_average,
        mass_error=mass_error,
        residual_grid=diagnostics["residual_grid"],
        residual_x=diagnostics["residual_x"],
        residual_tau=diagnostics["residual_tau"],
        history=np.array(history, dtype=object),
    )
    torch.save(model.state_dict(), config.output_dir / "solid_diffusion_pinn.pt")

    with (config.output_dir / "loss_history.csv").open("w", encoding="utf-8") as f:
        f.write("step,total,pde,initial,center,flux\n")
        for row in history:
            f.write(
                f"{row['step']:.0f},{row['loss']:.12e},{row['pde']:.12e},"
                f"{row['initial']:.12e},{row['center']:.12e},{row['flux']:.12e}\n"
            )

    plt.figure(figsize=(7.2, 4.8))
    plotted_reference = False
    for idx in np.linspace(0, len(tau) - 1, 5, dtype=int):
        plt.plot(x, c[idx], linewidth=2.0, label=f"PINN tau={tau[idx]:.2f}")
        ref_idx = int(round(idx * (len(tau_ref) - 1) / (len(tau) - 1)))
        ref_label = "finite-volume reference" if not plotted_reference else None
        plt.plot(x_ref, c_ref[ref_idx], "--", linewidth=1.7, color=plt.gca().lines[-1].get_color(), alpha=0.8, label=ref_label)
        plotted_reference = True
    plt.xlabel("Dimensionless particle radius x = r / R_s")
    plt.ylabel("Dimensionless concentration perturbation")
    plt.title("Solid-phase diffusion: PINN vs finite-volume reference")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(config.output_dir / "concentration_profiles.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    xx, tt = np.meshgrid(x, tau)
    contour = plt.contourf(xx, tt, c, levels=40, cmap="viridis")
    plt.colorbar(contour, label="Dimensionless concentration perturbation")
    plt.xlabel("x = r / R_s")
    plt.ylabel("tau = D_s t / R_s^2")
    plt.title("PINN concentration field")
    plt.tight_layout()
    plt.savefig(config.output_dir / "concentration_field.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    xx, tt = np.meshgrid(x, tau)
    max_abs = max(float(np.max(np.abs(error))), 1e-12)
    contour = plt.contourf(xx, tt, error, levels=41, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    plt.colorbar(contour, label="PINN - reference")
    plt.xlabel("x = r / R_s")
    plt.ylabel("tau = D_s t / R_s^2")
    plt.title("Pointwise error against finite-volume reference")
    plt.tight_layout()
    plt.savefig(config.output_dir / "error_field.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    residual = diagnostics["residual_grid"]
    rx = diagnostics["residual_x"]
    rt = diagnostics["residual_tau"]
    rxx, rtt = np.meshgrid(rx, rt)
    log_residual = np.log10(np.abs(residual) + 1e-10)
    contour = plt.contourf(rxx, rtt, log_residual, levels=41, cmap="magma")
    plt.colorbar(contour, label="log10(|PDE residual|)")
    plt.xlabel("x = r / R_s")
    plt.ylabel("tau = D_s t / R_s^2")
    plt.title("PINN PDE residual")
    plt.tight_layout()
    plt.savefig(config.output_dir / "pde_residual_field.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(tau, volume_average, linewidth=2.0, label="PINN volume average")
    plt.plot(tau, expected_average, "--", linewidth=1.8, label="Expected from flux balance")
    plt.xlabel("tau = D_s t / R_s^2")
    plt.ylabel("Volume-averaged concentration perturbation")
    plt.title("Mass conservation check")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.output_dir / "mass_conservation.png", dpi=220)
    plt.close()

    steps = np.array([row["step"] for row in history], dtype=float)
    plt.figure(figsize=(7.2, 4.2))
    for name in ["loss", "pde", "flux"]:
        values = np.array([row[name] for row in history], dtype=float)
        plt.semilogy(steps, values, marker="o", markersize=3, label=name)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Training loss history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.output_dir / "loss_history.png", dpi=220)
    plt.close()

    with (config.output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"RMSE against finite-volume reference: {rmse:.8e}\n")
        f.write(f"Max error against finite-volume reference: {max_error:.8e}\n")
        f.write(f"Mass balance RMS error: {float(np.sqrt(np.mean(mass_error**2))):.8e}\n")
        f.write(f"Mass balance max abs error: {float(np.max(np.abs(mass_error))):.8e}\n")
        f.write(f"Autograd PDE residual RMS: {diagnostics['pde_residual_rms']:.8e}\n")
        f.write(f"Autograd PDE residual max abs: {diagnostics['pde_residual_max_abs']:.8e}\n")
        f.write(f"Surface flux RMS error: {diagnostics['surface_flux_rms_error']:.8e}\n")
        f.write(f"Surface flux max abs error: {diagnostics['surface_flux_max_abs_error']:.8e}\n")
        f.write("Diagnostics excluding the incompatible startup corner tau < 1e-4:\n")
        f.write(f"Autograd PDE residual RMS, tau >= 1e-4: {diagnostics['pde_residual_rms_tau_ge_1e_4']:.8e}\n")
        f.write(f"Autograd PDE residual p99 abs, tau >= 1e-4: {diagnostics['pde_residual_p99_abs_tau_ge_1e_4']:.8e}\n")
        f.write(f"Autograd PDE residual max abs, tau >= 1e-4: {diagnostics['pde_residual_max_abs_tau_ge_1e_4']:.8e}\n")
        f.write(f"Surface flux RMS error, tau >= 1e-4: {diagnostics['surface_flux_rms_error_tau_ge_1e_4']:.8e}\n")
        f.write(f"Surface flux max abs error, tau >= 1e-4: {diagnostics['surface_flux_max_abs_error_tau_ge_1e_4']:.8e}\n")
        f.write(f"Initial condition RMS error: {diagnostics['initial_rms']:.8e}\n")
        f.write(f"Initial condition max abs error: {diagnostics['initial_max_abs']:.8e}\n")
        f.write(f"Center symmetry max abs slope: {diagnostics['center_slope_max_abs']:.8e}\n")
        f.write(f"Final total loss: {history[-1]['loss']:.8e}\n")
        f.write(f"Final PDE loss: {history[-1]['pde']:.8e}\n")
        f.write(f"Final initial loss: {history[-1]['initial']:.8e}\n")
        f.write(f"Final center BC loss: {history[-1]['center']:.8e}\n")
        f.write(f"Final flux BC loss: {history[-1]['flux']:.8e}\n")

    print(f"Saved results to {config.output_dir.resolve()}")
    print(f"RMSE against finite-volume reference: {rmse:.3e}")
    print(f"Max error against finite-volume reference: {max_error:.3e}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="PINN solver for P2D solid-phase spherical diffusion.")
    parser.add_argument("--tau-max", type=float, default=Config.tau_max)
    parser.add_argument("--phi", type=float, default=Config.phi)
    parser.add_argument("--adam-steps", type=int, default=Config.adam_steps)
    parser.add_argument("--lbfgs-steps", type=int, default=Config.lbfgs_steps)
    parser.add_argument("--n-collocation", type=int, default=Config.n_collocation)
    parser.add_argument("--n-boundary", type=int, default=Config.n_boundary)
    parser.add_argument("--hidden-width", type=int, default=Config.hidden_width)
    parser.add_argument("--hidden-layers", type=int, default=Config.hidden_layers)
    parser.add_argument("--output-dir", type=Path, default=Config.output_dir)
    args = parser.parse_args()
    return Config(
        tau_max=args.tau_max,
        phi=args.phi,
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        n_collocation=args.n_collocation,
        n_boundary=args.n_boundary,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        output_dir=args.output_dir,
    )


def main() -> None:
    config = parse_args()
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Solving with tau_max={config.tau_max}, surface flux phi={config.phi}")

    model = make_model(config).to(device)
    pts = sample_training_points(config, device)
    history = train(model, pts, config)
    save_outputs(model, history, config, device)


if __name__ == "__main__":
    main()
