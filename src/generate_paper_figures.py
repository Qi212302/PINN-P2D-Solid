"""Generate explanatory figures for the PINN solid-diffusion paper section."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from pinn_solid_diffusion import Config, grad, make_model, sample_training_points


torch.set_default_dtype(torch.float64)


RESULTS_DIR = Path("results")
FIGURE_DIR = RESULTS_DIR / "paper_figures"


def savefig(name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=240)
    plt.close()


def load_data() -> dict[str, np.ndarray]:
    return dict(np.load(RESULTS_DIR / "solid_diffusion_pinn_solution.npz", allow_pickle=True))


def load_model(config: Config, device: torch.device) -> torch.nn.Module:
    model = make_model(config).to(device)
    state = torch.load(RESULTS_DIR / "solid_diffusion_pinn.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def draw_box(ax: plt.Axes, xy: tuple[float, float], text: str, width: float = 1.45, height: float = 0.58) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.4,
        edgecolor="#2b2b2b",
        facecolor="#f6f7fb",
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=10)


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.4, color="#333333"))


def plot_physical_domain_schematic(config: Config) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    particle = Circle((0, 0), 1.0, facecolor="#eef5ff", edgecolor="#1f5f99", linewidth=2.0)
    ax.add_patch(particle)
    ax.plot([0, 1], [0, 0], color="#1f5f99", linewidth=2.2)
    ax.scatter([0, 1], [0, 0], color="#1f5f99", s=35, zorder=3)
    ax.text(-0.08, -0.13, "r = 0\nsymmetry", ha="right", va="top", fontsize=10)
    ax.text(1.05, -0.13, "r = R_s\nsurface flux", ha="left", va="top", fontsize=10)
    ax.text(0.48, 0.07, "solid diffusion\ninside particle", ha="center", va="bottom", fontsize=11)
    ax.annotate("", xy=(1.32, 0), xytext=(1.02, 0), arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#d62728"})
    ax.text(1.38, 0.0, r"$-D_s\partial c_s/\partial r=J_{surf}$", va="center", fontsize=11, color="#b01f1f")
    ax.annotate("", xy=(0.0, 0.32), xytext=(0.0, 0.03), arrowprops={"arrowstyle": "<->", "lw": 1.8, "color": "#2ca02c"})
    ax.text(-0.08, 0.36, r"$\partial c_s/\partial r=0$", ha="right", va="center", fontsize=11, color="#237a23")
    ax.text(0, -1.28, r"Dimensionless problem: $x=r/R_s,\ \tau=D_s t/R_s^2,\ \partial c/\partial x|_{x=1}=-\phi$" + f"\nHere, " + r"$\phi=$" + f"{config.phi:g}", ha="center", va="top", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.9)
    ax.set_ylim(-1.35, 1.2)
    ax.axis("off")
    ax.set_title("Spherical solid-particle diffusion domain and boundary conditions")
    savefig("01_physical_domain_boundary_conditions.png")


def plot_hard_constrained_architecture(config: Config) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    ax.set_xlim(0, 8.8)
    ax.set_ylim(0, 3.1)
    ax.axis("off")
    draw_box(ax, (0.2, 1.65), r"Inputs" + "\n" + r"$x,\tau$", 1.1, 0.68)
    draw_box(ax, (1.8, 1.65), "Feature map\n" + r"$z=2x^2-1$" + "\n" + r"$s=2\tau/\tau_{max}-1$", 1.7, 0.9)
    draw_box(ax, (4.05, 1.65), "MLP N(z,s)\n4 hidden layers\n64 tanh neurons", 1.65, 0.9)
    draw_box(ax, (6.28, 1.65), "Hard constraint\n" + r"$c_{PINN}=\tau N$", 1.55, 0.75)
    draw_box(ax, (7.95, 1.65), "Output\n" + r"$c(x,\tau)$", 0.7, 0.68)
    draw_arrow(ax, (1.3, 1.99), (1.78, 1.99))
    draw_arrow(ax, (3.52, 1.99), (4.03, 1.99))
    draw_arrow(ax, (5.72, 1.99), (6.26, 1.99))
    draw_arrow(ax, (7.84, 1.99), (7.93, 1.99))
    ax.text(4.45, 0.72, r"Exactly enforces: $c(x,0)=0$ and $\partial c/\partial x(0,\tau)=0$", ha="center", fontsize=11, color="#1f5f99")
    ax.text(4.45, 0.34, r"Trainable residual terms: PDE residual and surface flux boundary residual", ha="center", fontsize=10, color="#444444")
    ax.set_title("Hard-constrained PINN ansatz for solid-phase diffusion")
    savefig("02_hard_constrained_pinn_architecture.png")


def plot_training_points(config: Config, device: torch.device) -> None:
    pts = sample_training_points(config, device)
    x_f = pts["x_f"].detach().cpu().numpy().ravel()
    tau_f = pts["tau_f"].detach().cpu().numpy().ravel()
    x_i = pts["x_i"].detach().cpu().numpy().ravel()
    tau_i = pts["tau_i"].detach().cpu().numpy().ravel()
    tau_b = pts["tau_b"].detach().cpu().numpy().ravel()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    rng = np.random.default_rng(config.seed)
    keep = rng.choice(len(x_f), size=min(1800, len(x_f)), replace=False)
    ax.scatter(x_f[keep], tau_f[keep], s=7, alpha=0.22, color="#1f77b4", label="PDE collocation")
    ax.scatter(x_i, tau_i, s=8, alpha=0.45, color="#2ca02c", label="initial points")
    ax.scatter(np.ones_like(tau_b), tau_b, s=6, alpha=0.25, color="#d62728", label="surface-flux BC")
    ax.scatter(np.zeros_like(tau_b), tau_b, s=6, alpha=0.25, color="#9467bd", label="center-symmetry check")
    ax.set_xlabel(r"$x=r/R_s$")
    ax.set_ylabel(r"$\tau=D_s t/R_s^2$")
    ax.set_title("Training point distribution in the dimensionless space-time domain")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(-0.01, config.tau_max + 0.01)
    savefig("03_training_points_distribution.png")


def plot_error_profiles(data: dict[str, np.ndarray]) -> None:
    x = data["x"]
    tau = data["tau"]
    error = data["error"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for target in [0.1, 0.2, 0.3, 0.4]:
        idx = int(np.argmin(np.abs(tau - target)))
        ax.plot(x, error[idx], linewidth=2.0, label=fr"$\tau={tau[idx]:.2f}$")
    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_xlabel(r"$x=r/R_s$")
    ax.set_ylabel(r"$c_{PINN}-c_{ref}$")
    ax.set_title("Radial error profiles at selected times")
    ax.legend()
    savefig("04_error_profiles_selected_times.png")


def plot_center_surface_concentration(data: dict[str, np.ndarray]) -> None:
    x = data["x"]
    tau = data["tau"]
    c = data["c"]
    center = c[:, 0]
    surface = c[:, -1]
    average = data["volume_average"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(tau, center, linewidth=2.2, label="center concentration")
    ax.plot(tau, surface, linewidth=2.2, label="surface concentration")
    ax.plot(tau, average, "--", linewidth=1.8, label="volume average")
    ax.fill_between(tau, surface, center, alpha=0.14, color="#1f77b4", label="radial diffusion lag")
    ax.set_xlabel(r"$\tau=D_s t/R_s^2$")
    ax.set_ylabel("Dimensionless concentration perturbation")
    ax.set_title("Center and surface concentration evolution")
    ax.legend()
    ax.set_xlim(tau[0], tau[-1])
    savefig("05_center_surface_concentration_evolution.png")


def surface_flux_from_model(model: torch.nn.Module, config: Config, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    tau = np.linspace(0.0, config.tau_max, 401)
    tau_t = torch.tensor(tau[:, None], device=device, requires_grad=True)
    x_t = torch.ones_like(tau_t, requires_grad=True)
    c = model(torch.cat([x_t, tau_t], dim=1))
    slope = grad(c, x_t).detach().cpu().numpy().ravel()
    return tau, slope


def plot_surface_flux_validation(model: torch.nn.Module, config: Config, device: torch.device) -> None:
    tau, slope = surface_flux_from_model(model, config, device)
    target = -config.phi * np.ones_like(tau)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True, height_ratios=[2.0, 1.0])
    axes[0].plot(tau, slope, linewidth=2.0, label=r"PINN $\partial c/\partial x|_{x=1}$")
    axes[0].plot(tau, target, "--", linewidth=1.8, label=fr"target $-\phi={-config.phi:g}$")
    axes[0].set_ylabel("Surface gradient")
    axes[0].set_title("Surface flux boundary validation")
    axes[0].legend()
    axes[1].plot(tau, slope - target, linewidth=1.8, color="#d62728")
    axes[1].axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[1].axvspan(0.0, 1e-4, color="#999999", alpha=0.18, label=r"startup corner")
    axes[1].set_xlabel(r"$\tau=D_s t/R_s^2$")
    axes[1].set_ylabel("Error")
    axes[1].legend(loc="upper right", fontsize=9)
    savefig("06_surface_flux_boundary_validation.png")

    mask = tau >= 1e-4
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True, height_ratios=[2.0, 1.0])
    axes[0].plot(tau[mask], slope[mask], linewidth=2.0, label=r"PINN $\partial c/\partial x|_{x=1}$")
    axes[0].plot(tau[mask], target[mask], "--", linewidth=1.8, label=fr"target $-\phi={-config.phi:g}$")
    axes[0].set_ylabel("Surface gradient")
    axes[0].set_title(r"Surface flux validation after startup corner ($\tau \geq 10^{-4}$)")
    axes[0].legend()
    error = slope - target
    axes[1].plot(tau[mask], error[mask], linewidth=1.8, color="#d62728")
    axes[1].axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel(r"$\tau=D_s t/R_s^2$")
    axes[1].set_ylabel("Error")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    savefig("06b_surface_flux_boundary_validation_zoom.png")


def plot_volume_average_error(data: dict[str, np.ndarray]) -> None:
    tau = data["tau"]
    mass_error = data["mass_error"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(tau, mass_error, linewidth=2.0, color="#d62728")
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\tau=D_s t/R_s^2$")
    ax.set_ylabel("Volume-average error")
    ax.set_title("Mass conservation error")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.set_xlim(tau[0], tau[-1])
    savefig("07_volume_average_mass_error.png")


def plot_residual_statistics(data: dict[str, np.ndarray]) -> None:
    residual = data["residual_grid"]
    x = data["residual_x"]
    tau = data["residual_tau"]
    rms_tau = np.sqrt(np.mean(residual**2, axis=1))
    p99_tau = np.percentile(np.abs(residual), 99.0, axis=1)
    rms_x = np.sqrt(np.mean(residual**2, axis=0))
    p99_x = np.percentile(np.abs(residual), 99.0, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    axes[0].semilogy(tau, rms_tau, linewidth=2.0, label="RMS residual")
    axes[0].semilogy(tau, p99_tau, "--", linewidth=1.7, label="99th percentile")
    axes[0].axvspan(0.0, 1e-4, color="#999999", alpha=0.18)
    axes[0].set_xlabel(r"$\tau=D_s t/R_s^2$")
    axes[0].set_ylabel("Residual magnitude")
    axes[0].set_title("Residual statistics vs time")
    axes[0].legend(fontsize=9)

    axes[1].semilogy(x, rms_x, linewidth=2.0, label="RMS residual")
    axes[1].semilogy(x, p99_x, "--", linewidth=1.7, label="99th percentile")
    axes[1].set_xlabel(r"$x=r/R_s$")
    axes[1].set_ylabel("Residual magnitude")
    axes[1].set_title("Residual statistics vs radius")
    axes[1].legend(fontsize=9)
    savefig("08_pde_residual_statistics.png")


def write_figure_index() -> None:
    descriptions = [
        ("01_physical_domain_boundary_conditions.png", "Spherical particle domain, center symmetry, and surface flux boundary."),
        ("02_hard_constrained_pinn_architecture.png", "Hard-constrained PINN ansatz and its exact IC/center-BC enforcement."),
        ("03_training_points_distribution.png", "PDE, initial, center, and surface-boundary training points."),
        ("04_error_profiles_selected_times.png", "Radial PINN-reference error curves at selected times."),
        ("05_center_surface_concentration_evolution.png", "Center, surface, and volume-averaged concentration evolution."),
        ("06_surface_flux_boundary_validation.png", "Surface gradient compared with the imposed flux boundary condition."),
        ("06b_surface_flux_boundary_validation_zoom.png", "Post-startup zoom of the surface flux boundary validation."),
        ("07_volume_average_mass_error.png", "Mass-conservation error in volume-averaged concentration."),
        ("08_pde_residual_statistics.png", "RMS and 99th-percentile PDE residual statistics versus time and radius."),
        ("09_flux_weight_sensitivity_metrics.png", "Sensitivity of RMSE, flux error, PDE loss, and mass error to the flux-loss weight."),
        ("10_flux_weight_constraint_balance.png", "Relative balance between reference error, PDE loss, and flux-boundary error."),
    ]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    with (FIGURE_DIR / "figure_index.md").open("w", encoding="utf-8") as f:
        f.write("# Paper figure index\n\n")
        for name, desc in descriptions:
            f.write(f"- `{name}`: {desc}\n")


def main() -> None:
    config = Config(
        output_dir=RESULTS_DIR,
        hidden_width=64,
        hidden_layers=4,
        n_collocation=3500,
        n_boundary=800,
        adam_steps=1500,
        lbfgs_steps=350,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data()
    model = load_model(config, device)

    plot_physical_domain_schematic(config)
    plot_hard_constrained_architecture(config)
    plot_training_points(config, device)
    plot_error_profiles(data)
    plot_center_surface_concentration(data)
    plot_surface_flux_validation(model, config, device)
    plot_volume_average_error(data)
    plot_residual_statistics(data)
    write_figure_index()
    print(f"Saved paper figures to {FIGURE_DIR.resolve()}")


if __name__ == "__main__":
    main()
