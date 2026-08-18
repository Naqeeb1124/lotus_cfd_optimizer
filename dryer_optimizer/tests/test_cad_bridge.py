"""Tests for the corrected Y-Z optimizer to CAD dimensional contract."""

import numpy as np

from dryer_optimizer.sldw_optimized import CadDimensions, _contiguous_runs


def test_corrected_optimizer_domain_matches_cad() -> None:
    dimensions = CadDimensions()
    assert np.isclose(dimensions.optimizer_domain_width, 702.0)
    assert np.isclose(dimensions.optimizer_domain_height, 1630.0)
    assert np.isclose(dimensions.chamber_width, 484.0)
    assert np.isclose(dimensions.chamber_height, 1630.0)


def test_topology_runs_are_half_open() -> None:
    row = np.array([False, True, True, False, True, False, True, True, True])
    assert _contiguous_runs(row) == [(1, 3), (4, 5), (6, 9)]
