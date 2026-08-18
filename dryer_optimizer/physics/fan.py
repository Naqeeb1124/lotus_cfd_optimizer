"""Fan performance-curve utilities for the internal actuator model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FanCurve:
    """Static pressure rise as a function of 3D volumetric flow rate."""

    flow_points: tuple[float, ...]
    pressure_points: tuple[float, ...]

    @classmethod
    def from_pairs(cls, pairs: tuple[tuple[float, float], ...]) -> "FanCurve":
        flows = tuple(float(pair[0]) for pair in pairs)
        pressures = tuple(float(pair[1]) for pair in pairs)
        if len(flows) < 2 or any(b <= a for a, b in zip(flows, flows[1:])):
            raise ValueError("Fan flow points must be strictly increasing.")
        if any(not np.isfinite(p) or p < 0.0 for p in pressures):
            raise ValueError("Fan pressures must be finite and non-negative.")
        if any(b > a for a, b in zip(pressures, pressures[1:])):
            raise ValueError("Fan static pressure must be monotonically non-increasing with flow.")
        return cls(flows, pressures)

    @property
    def maximum_flow(self) -> float:
        return self.flow_points[-1]

    @property
    def shutoff_pressure(self) -> float:
        return self.pressure_points[0]

    def pressure_and_slope(self, flow_rate: float) -> tuple[float, float]:
        """Return clipped pressure and exact segment slope ``dP/dQ``."""
        q = float(flow_rate)
        if q <= self.flow_points[0]:
            slope = (self.pressure_points[1] - self.pressure_points[0]) / (self.flow_points[1] - self.flow_points[0])
            return self.pressure_points[0], slope if q >= 0 else 0.0
        if q >= self.flow_points[-1]:
            slope = (self.pressure_points[-1] - self.pressure_points[-2]) / (self.flow_points[-1] - self.flow_points[-2])
            return self.pressure_points[-1], 0.0
        index = int(np.searchsorted(self.flow_points, q) - 1)
        q0, q1 = self.flow_points[index], self.flow_points[index + 1]
        p0, p1 = self.pressure_points[index], self.pressure_points[index + 1]
        slope = (p1 - p0) / (q1 - q0)
        return p0 + slope * (q - q0), slope

    def pressure(self, flow_rate: float) -> float:
        return self.pressure_and_slope(flow_rate)[0]

    def slope(self, flow_rate: float) -> float:
        return self.pressure_and_slope(flow_rate)[1]
