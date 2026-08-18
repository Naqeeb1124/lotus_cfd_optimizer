"""Visualization helpers for optimizer diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dryer_optimizer.geometry.domain import DryerGeometry
from dryer_optimizer.optimization.objective import ObjectiveValue
from dryer_optimizer.physics.solver import FlowState


def plot_flow(
    state: FlowState,
    geometry: DryerGeometry,
    objective: ObjectiveValue,
    path: str | Path,
) -> Path:
    """Save density, speed, and pressure diagnostics."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    speed = np.sqrt(
        0.25 * (state.u[:, :-1] + state.u[:, 1:]) ** 2
        + 0.25 * (state.v[:-1, :] + state.v[1:, :]) ** 2
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    extent = (0, geometry.grid.width, 0, geometry.grid.height)
    axes[0].imshow(state.density, origin="lower", extent=extent, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    axes[0].set_title("Physical density / baffles")
    axes[1].imshow(speed, origin="lower", extent=extent, aspect="auto", cmap="magma")
    axes[1].set_title("Velocity magnitude")
    pressure = axes[2].imshow(state.p, origin="lower", extent=extent, aspect="auto", cmap="coolwarm")
    axes[2].set_title(
        f"Pressure | fan={objective.fan_pressure:.3g} Pa, Q={objective.fan_flow:.3g} m³/s"
    )
    for ax in axes:
        for elevation in [geometry.grid.cell_y[row] for row in geometry.tray_cell_rows]:
            ax.axhline(elevation, color="cyan", linewidth=0.35, alpha=0.7)
        ax.set_xlabel("airflow depth / horizontal domain [m]")
        ax.set_ylabel("height [m]")
    fig.colorbar(pressure, ax=axes[2], shrink=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_optimized_distribution(
    density: np.ndarray,
    binary: np.ndarray,
    tray_averages: np.ndarray,
    *,
    width: float,
    height: float,
    tray_elevations: np.ndarray,
    path: str | Path,
    velocity_u: np.ndarray | None = None,
    velocity_v: np.ndarray | None = None,
    objective: float | None = None,
    horizontal_label: str = "horizontal domain [m]",
    vertical_label: str = "height [m]",
) -> Path:
    """Plot the optimized topology beside its tray-by-tray distribution.

    This is intentionally independent of build123d so it can be used on a
    headless machine and still show the actual optimizer field before CAD.
    """
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    density = np.asarray(density, dtype=float)
    binary = np.asarray(binary, dtype=bool)
    tray_averages = np.asarray(tray_averages, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    extent = (0, width, 0, height)
    axes[0, 0].imshow(density, origin="lower", extent=extent, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    if np.any(binary) and not np.all(binary):
        axes[0, 0].contour(binary.astype(float), levels=[0.5], extent=extent, origin="lower", colors="#e65100", linewidths=0.7)
    axes[0, 0].set_title("Optimized physical density and extracted baffles")
    axes[1, 0].imshow(binary, origin="lower", extent=extent, aspect="auto", cmap="Greys")
    axes[1, 0].set_title("Binary CAD topology")
    for ax in (axes[0, 0], axes[1, 0]):
        for elevation in tray_elevations:
            ax.axhline(elevation, color="#00acc1", linewidth=0.35, alpha=0.65)
        ax.set_xlabel(horizontal_label)
        ax.set_ylabel(vertical_label)

    indices = np.arange(1, len(tray_averages) + 1)
    mean_velocity = float(np.mean(tray_averages))
    cv = float(np.std(tray_averages) / max(abs(mean_velocity), 1.0e-12))
    axes[0, 1].bar(indices, tray_averages, color="#1565c0")
    axes[0, 1].axhline(mean_velocity, color="#e65100", linestyle="--", label=f"mean={mean_velocity:.4g}")
    axes[0, 1].set_title(f"Tray normal velocity distribution (CV={cv:.4g})")
    axes[0, 1].set_xlabel("tray index")
    axes[0, 1].set_ylabel("normal velocity [m/s]")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.25)

    if velocity_u is not None and velocity_v is not None:
        velocity_u = np.asarray(velocity_u, dtype=float)
        velocity_v = np.asarray(velocity_v, dtype=float)
        speed = np.sqrt(
            0.25 * (velocity_u[:, :-1] + velocity_u[:, 1:]) ** 2
            + 0.25 * (velocity_v[:-1, :] + velocity_v[1:, :]) ** 2
        )
        image = axes[1, 1].imshow(speed, origin="lower", extent=extent, aspect="auto", cmap="magma")
        fig.colorbar(image, ax=axes[1, 1], label="speed [m/s]")
        axes[1, 1].set_title("Actual MAC-solved flow speed")
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "Flow field not stored in topology.npz", ha="center", va="center")
    if objective is not None:
        fig.suptitle(f"Dryer topology distribution | objective={objective:.5g}")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_history(history, path: str | Path) -> Path:
    """Save objective and material-fraction convergence curves."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    iterations = np.arange(len(history.objective))
    ax.plot(iterations, history.objective, label="objective", linewidth=2)
    ax.plot(iterations, history.cv, label="tray CV")
    ax.plot(iterations, history.design_fraction, label="solid fraction")
    if getattr(history, "fan_pressure", None):
        ax2 = ax.twinx()
        ax2.plot(iterations, history.fan_pressure, color="#c62828", linestyle="--", label="fan pressure [Pa]")
        ax2.set_ylabel("fan pressure [Pa]")
        ax2.axhline(300.0, color="#c62828", alpha=0.35, linewidth=0.8)
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
