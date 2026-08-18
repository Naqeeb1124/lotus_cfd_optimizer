# Dryer Flow Topology Optimizer

**Goal:** Generate a manufacturable 2D internal-flow topology that improves
airflow uniformity across exactly 20 drying trays while controlling baffle
material and fan pressure demand.

**Current method:** Full-height 2D Y-Z side section + steady incompressible
Navier–Stokes/Brinkman flow + internal fan actuator + discrete adjoint
sensitivity + density filtering/projection + explicit pressure and volume
constraints.

**Output:** A 2D optimized baffle/flow-distribution topology suitable for CAD
reconstruction and later 3D CFD validation. This is a reduced engineering
model, not a replacement for the final 3D Fluent/FloEFD case.

## Model dimensions

Defaults mirror `../test_files/sldw_cad.py`:

- exactly 20 trays;
- tray width: 400 mm;
- tray depth: 300 mm;
- tray wall height: 36.5 mm;
- tray air gap: 25 mm;
- pitch: 61.5 mm;
- bottom clearance: 50 mm;
- mechanical room: 350 mm;
- full cabinet/optimizer height: **1.630 m**;
- side-section airflow depth: **0.702 m**;
- 315 mm plug-fan actuator in the mechanical-room region.

The optimizer coordinate system is horizontal CAD Y airflow depth and vertical
CAD Z height from the cabinet bottom. The CAD bridge extrudes binary Y-Z
baffles across the cabinet X width.

## Fan model

The supplied fan curve is stored in `PhysicsConfig` and used directly:

```text
Volume flow Q [m³/s] : 0.00, 0.25, 0.50, 0.75, 1.00
Static pressure rise [Pa]: 1800, 1650, 1300, 700, 0
```

The fan is an internal finite-thickness actuator, not a prescribed flat
velocity inlet. On the MAC grid:

```text
Q_fan = fan_source_direction * b * sum(u_fan_faces * dy)
body_force_x = fan_source_direction * DeltaP(Q_fan) / fan_thickness
```

where `b = 0.315 m` is the specified out-of-plane fan width and
`fan_source_direction = -1.0` is the sign factor that makes positive `Q_fan`
mean air flowing in the correct physical direction (supply -> trays -> return).
The pressure curve is piecewise linear, clipped at the supplied endpoints, and
its segment slope is included in the final Jacobian. This is the correct
reduced 2D use of a 3D fan curve for a planar slice, subject to later 3D
calibration. Fan swirl, blade passing, stall, and radial discharge structure
are not represented.

The model is a SEALED box: all faces are no-slip walls and the internal fan is
the only momentum source, recirculating air through the tray stack. The global
pressure null space is anchored at `p = 0` gauge in the four wall corners and
in a fluid cell at the center of the mechanical room (row `ny - ny//8`,
column `nx//2`), guaranteed to avoid the walls, tray floors, and fan actuator.

## Governing equations

The steady incompressible equations are discretized on a staggered MAC grid:

```text
rho (u · grad) u = -grad(p) + mu_eff Laplacian(u) - alpha(rho_s) u + f_fan
div(u) = 0
```

The nonlinear convective term uses first-order upwind derivatives and damped
Newton iterations with residual backtracking and continuation between topology
iterations. The effective viscosity is

```text
mu_eff = molecular_viscosity + eddy_viscosity
       = 1.8e-5 + 2.0e-4 Pa s by default
```

The elevated viscosity is an explicit reduced-model closure for unresolved
mixing; it is not a calibrated turbulence model and must be calibrated against
3D CFD or experiment.

The optimization variable stored as `density` is **solid fraction**
`rho_s = 1 - gamma`, where gamma is fluid fraction. Therefore 0 means fluid
and 1 means baffle material. The Brinkman law is

```text
alpha = alpha_min + (alpha_max-alpha_min) * q*rho_s/(q + 1-rho_s)
```

with `q=0.05` and `alpha_max=1e5` by default. Geometry-locked solids override
the density field with `alpha_max` and are never design variables.

## Objective and constraints

For each of the 20 tray sample lines, the optimizer computes an average
horizontal tray velocity `v_i`. The target is derived from the SIGNED fan flow
(the actual internal-fan throughput) divided among 20 tray lines and converted
to a velocity using the fan's out-of-plane width and the tray span:

```text
v_target = Q_fan / (20 * fan_span * span),  span = max(tray_depth, dx)
J_uniformity = mean((v_i - v_target)^2) / velocity_reference^2
```

The signed fan flow (not `abs(Q_fan)`) keeps `dQ/du_face = fan_source_direction * b * dy`
continuous through `Q = 0`, preserving adjoint differentiability at the shutoff
operating point.

The optimization objective also contains:

- material cost proportional to solid fraction;
- a quadratic penalty outside the 5–15% allowed solid-volume band;
- a quadratic penalty when fan pressure exceeds the 300 Pa default limit.

This prevents the optimizer from creating pinholes or an almost-solid plug just
to equalize the tray samples.

## Adjoint verification

At a converged nonlinear state, the solver stores the analytic Jacobian of the
full residual, including upwind convection and `dDeltaP/dQ` fan coupling. The
adjoint solves

```text
(J_state)^T lambda = dJ/dstate
```

and evaluates the Brinkman density derivative

```text
dJ/drho = dJ_explicit/drho - lambda^T (dR/drho).
```

The state gradient includes the fan-flow-dependent target and pressure-cap
chain rules, both scaled by `dQ/du_face = fan_source_direction * b * dy`.
`fan_source_direction` is owned by `PhysicsConfig` and threaded through the
adjoint into the objective gradient (it is a required keyword, so a non-default
direction cannot silently produce a wrong-signed gradient). The test suite
compares the derivative with central finite differences on a small grid,
including the `Q=0` startup state, before accepting the implementation.

## Running

Use the project virtual environment on Windows:

```text
.venv\\Scripts\\activate
python -m pytest dryer_optimizer/tests -q
python -m dryer_optimizer.main --iterations 5 --nx 24 --ny 56
```

The default 48x112 grid is more expensive because each nonlinear solve uses
sparse Newton factorizations. Use 24x56 for fast iteration while validating
topology trends, then increase resolution for the final candidate:

```text
python -m dryer_optimizer.main --iterations 30 --nx 48 --ny 112
```

Outputs are written to `dryer_optimizer/data/output/`:

- `topology.npz`: density, binary topology, MAC velocities, pressure, tray
  values, objective, fan flow, fan pressure, and target velocity;
- `topology.svg`: binary contour-style topology;
- `flow.png`: density, actual MAC speed, pressure, tray elevations, and fan
  operating-point annotation;
- `history.png`: objective/CV/solid fraction and fan-pressure history;
- `summary.json`: dimensions, fan curve, constraints, and final metrics.

## CAD bridge

After generating topology:

```text
python dryer_optimizer/sldw_optimized.py --density-threshold 0.10
python dryer_optimizer/sldw_optimized.py --view --density-threshold 0.10
```

The explicit threshold is only a provisional CAD conversion of the continuous
field; it does not modify the optimizer NPZ. STEP and visualization files are
written to `dryer_optimizer/sldw_dump/`.

## Scope and limitations

The model intentionally does not claim turbulent fidelity, fan swirl, heat or
mass transfer, humidity, evaporation, transient behavior, or 3D optimality.
Tray perforation is represented by a reduced tray resistance and the supplied
CAD tray floors still need porous/perforated treatment in a CFD comparison.
Manufacturing cost is currently represented by solid volume, filtering, minimum
feature scale, and pressure constraints; a real quotation model can be added
after the baseline/optimized 3D CFD comparison.
