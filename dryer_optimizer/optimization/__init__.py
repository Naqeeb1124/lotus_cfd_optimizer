"""Objective, adjoint, filtering, and density-update utilities."""

from .adjoint import AdjointResult, compute_adjoint
from .filters import DensityFilter, smooth_projection
from .objective import (
    ObjectiveValue,
    evaluate_objective,
    objective_density_gradient,
    objective_state_gradient,
)
from .update import (
    OptimizationHistory,
    initial_density,
    physical_density,
    update_density,
)

__all__ = [
    "AdjointResult",
    "DensityFilter",
    "ObjectiveValue",
    "OptimizationHistory",
    "compute_adjoint",
    "evaluate_objective",
    "initial_density",
    "objective_density_gradient",
    "objective_state_gradient",
    "physical_density",
    "smooth_projection",
    "update_density",
]
