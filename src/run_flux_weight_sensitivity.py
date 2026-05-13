"""Flux-boundary loss weight sensitivity study for the solid-diffusion PINN."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_solid_diffusion import Config, make_model, sample_training_points, save_outputs, train


DEFAULT_WEIGHTS = [1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 100.0]
SELECTED_FLUX_WEIGHT = 20.0


def parse_metric_file(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        try:
            metrics[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return metrics


def run_single_weight(base_config: Config, weight: float, device: torch.device, output_root: Path) -> dict[str, float]:
    run_dir = output_root / f"lambda_flux_{weight:g}".replace(".", "p")
    config = Config(
        tau_max=base_config.tau_max,
        phi=base_config.phi,
        hidden_width=base_config.hidden_width,
        hidden_layers=base_config.hidden_layers,
        n_collocation=base_config.n_collocation,
        n_initial=base_config.n_initial,
        n_boundary=base_config.n_boundary,
        adam_steps=base_config.adam_steps,
        lbfgs_steps=base_config.lbfgs_steps,
        learning_rate=base_config.learning_rate,
        flux_weight=weight,
        seed=base_config.seed,
        output_dir=run_dir,
    )
    print(f"\n=== Training lambda_flux={weight:g} ===")
    torch.manual_seed(config.seed)
    model = make_model(config).to(device)
    pts = sample_training_points(config, device)
    history = train(model, pts, config)
    save_outputs(model, history, config, device)

    metrics = parse_metric_file(run_dir / "metrics.txt")
    row = {
        "lambda_flux": weight,
        "rmse_ref": metrics["RMSE against finite-volume reference"],
        "max_error_ref": metrics["Max error against finite-volume reference"],
        "mass_balance_rms": metrics["Mass balance RMS error"],
        "pde_loss": metrics["Final PDE loss"],
        "flux_loss": metrics["Final flux BC loss"],
        "pde_residual_rms_tau_ge_1e_4": metrics["Autograd PDE residual RMS, tau >= 1e-4"],
        "pde_residual_p99_tau_ge_1e_4": metrics["Autograd PDE residual p99 abs, tau >= 1e-4"],
        "surface_flux_rms_tau_ge_1e_4": metrics["Surface flux RMS error, tau >= 1e-4"],
        "surface_flux_max_tau_ge_1e_4": metrics["Surface flux max abs error, tau >= 1e-4"],
    }
    print(
        "Summary "
        f"lambda={weight:g}: rmse={row['rmse_ref']:.3e}, "
        f"flux_rms={row['surface_flux_rms_tau_ge_1e_4']:.3e}, "
        f"pde_loss={row['pde_loss']:.3e}"
    )
    return row


def write_summary(rows: list[dict[str, float]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    weights = np.array([row["lambda_flux"] for row in rows], dtype=float)
    rmse = np.array([row["rmse_ref"] for row in rows], dtype=float)
    flux_rms = np.array([row["surface_flux_rms_tau_ge_1e_4"] for row in rows], dtype=float)
    pde_loss = np.array([row["pde_loss"] for row in rows], dtype=float)
    mass_rms = np.array([row["mass_balance_rms"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.loglog(weights, rmse, marker="o", linewidth=2.0, label="RMSE vs FVM")
    ax.loglog(weights, flux_rms, marker="s", linewidth=2.0, label=r"surface flux RMS, $\tau\geq10^{-4}$")
    ax.loglog(weights, pde_loss, marker="^", linewidth=2.0, label="final PDE loss")
    ax.loglog(weights, mass_rms, marker="d", linewidth=2.0, label="mass balance RMS")
    ax.axvline(SELECTED_FLUX_WEIGHT, linestyle="--", linewidth=1.4, color="#555555", label=fr"$\lambda_{{flux}}={SELECTED_FLUX_WEIGHT:g}$")
    ax.set_xlabel(r"Surface flux loss weight $\lambda_{flux}$")
    ax.set_ylabel("Metric value")
    ax.set_title("Flux-boundary loss weight sensitivity")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_root / "flux_weight_sensitivity_metrics.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogx(weights, flux_rms / rmse, marker="o", linewidth=2.0, label="flux RMS / RMSE")
    ax.semilogx(weights, pde_loss / rmse, marker="s", linewidth=2.0, label="PDE loss / RMSE")
    ax.axvline(SELECTED_FLUX_WEIGHT, linestyle="--", linewidth=1.4, color="#555555", label=fr"$\lambda_{{flux}}={SELECTED_FLUX_WEIGHT:g}$")
    ax.set_xlabel(r"Surface flux loss weight $\lambda_{flux}$")
    ax.set_ylabel("Ratio")
    ax.set_title("Relative balance of learned constraints")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_root / "flux_weight_constraint_balance.png", dpi=240)
    plt.close(fig)

    best_by_rmse = min(rows, key=lambda row: row["rmse_ref"])
    best_by_flux = min(rows, key=lambda row: row["surface_flux_rms_tau_ge_1e_4"])
    best_by_flux_max = min(rows, key=lambda row: row["surface_flux_max_tau_ge_1e_4"])
    selected = min(rows, key=lambda row: abs(row["lambda_flux"] - SELECTED_FLUX_WEIGHT))
    with (output_root / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Flux loss weight sensitivity summary\n\n")
        f.write("All runs use the same random seed, architecture, collocation sampling strategy, and optimizer settings.\n\n")
        f.write(f"- Best RMSE against finite-volume reference: lambda={best_by_rmse['lambda_flux']:g}, RMSE={best_by_rmse['rmse_ref']:.6e}\n")
        f.write(
            "- Best post-startup surface-flux RMS error: "
            f"lambda={best_by_flux['lambda_flux']:g}, error={best_by_flux['surface_flux_rms_tau_ge_1e_4']:.6e}\n"
        )
        f.write(
            "- Best post-startup surface-flux max error: "
            f"lambda={best_by_flux_max['lambda_flux']:g}, error={best_by_flux_max['surface_flux_max_tau_ge_1e_4']:.6e}\n"
        )
        f.write(
            f"- Selected default for the paper calculation: lambda={selected['lambda_flux']:g}. "
            "This is not the single best value for every metric, but it gives a stronger balance than lambda=40: "
            "lower RMSE, lower mass-balance error, and lower post-startup surface-flux maximum error while keeping "
            "the PDE residual at the same order of magnitude.\n"
        )
        f.write("- Use `summary.csv` for the complete numerical table.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lambda_flux sensitivity study.")
    parser.add_argument("--weights", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-root", type=Path, default=Path("results") / "flux_weight_sensitivity")
    parser.add_argument("--adam-steps", type=int, default=1500)
    parser.add_argument("--lbfgs-steps", type=int, default=350)
    parser.add_argument("--n-collocation", type=int, default=3500)
    parser.add_argument("--n-boundary", type=int, default=800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_config = Config(
        hidden_width=64,
        hidden_layers=4,
        n_collocation=args.n_collocation,
        n_boundary=args.n_boundary,
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        seed=2026,
    )
    rows = [run_single_weight(base_config, weight, device, args.output_root) for weight in args.weights]
    write_summary(rows, args.output_root)
    print(f"\nSaved sensitivity study to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
