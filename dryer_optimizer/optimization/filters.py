"""Density filtering and projection for manufacturable topology updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class DensityFilter:
    """Symmetric-neighborhood density filter with an exact transpose action."""

    shape: tuple[int, int]
    radius_cells: int

    def __post_init__(self) -> None:
        if self.radius_cells < 0:
            raise ValueError("radius_cells cannot be negative.")
        self._matrix = self._build_matrix()

    @property
    def matrix(self) -> csr_matrix:
        return self._matrix

    def _build_matrix(self) -> csr_matrix:
        ny, nx = self.shape
        if self.radius_cells == 0:
            return csr_matrix(np.eye(ny * nx, dtype=float))
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        radius = self.radius_cells
        for row in range(ny):
            for col in range(nx):
                index = row * nx + col
                neighbors = [
                    (nr, nc)
                    for nr in range(max(0, row - radius), min(ny, row + radius + 1))
                    for nc in range(max(0, col - radius), min(nx, col + radius + 1))
                ]
                weight = 1.0 / len(neighbors)
                for nr, nc in neighbors:
                    rows.append(index)
                    cols.append(nr * nx + nc)
                    values.append(weight)
        return csr_matrix((values, (rows, cols)), shape=(ny * nx, ny * nx))

    def apply(self, density: np.ndarray) -> np.ndarray:
        field = np.asarray(density, dtype=float)
        if field.shape != self.shape:
            raise ValueError(f"density has shape {field.shape}, expected {self.shape}.")
        return np.asarray(self._matrix @ field.ravel()).reshape(self.shape)

    def transpose_apply(self, gradient: np.ndarray) -> np.ndarray:
        field = np.asarray(gradient, dtype=float)
        if field.shape != self.shape:
            raise ValueError(f"gradient has shape {field.shape}, expected {self.shape}.")
        return np.asarray(self._matrix.transpose() @ field.ravel()).reshape(self.shape)


def smooth_projection(
    filtered_density: np.ndarray,
    *,
    threshold: float = 0.5,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return smooth Heaviside projection and its pointwise derivative."""
    if not 0 < threshold < 1 or beta <= 0:
        raise ValueError("threshold must be in (0, 1) and beta must be positive.")
    rho = np.asarray(filtered_density, dtype=float)
    denominator = np.tanh(beta * threshold) + np.tanh(beta * (1.0 - threshold))
    projected = (
        np.tanh(beta * threshold) + np.tanh(beta * (rho - threshold))
    ) / denominator
    derivative = beta * (1.0 - np.tanh(beta * (rho - threshold)) ** 2) / denominator
    return np.clip(projected, 0.0, 1.0), derivative
