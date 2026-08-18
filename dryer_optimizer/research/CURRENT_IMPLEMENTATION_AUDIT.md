# CURRENT_IMPLEMENTATION_AUDIT

Audit of the existing production code under `dryer_optimizer/` (physics, optimization, geometry,
visualization, tests, config, main, CAD bridge). Reference implementations live only under
`research/reference_code/`; the verified source-level analysis is in `research/PIPELINE_TRACE.md`.

**Headline facts (measured 2026-08-13):**
- Test suite: **11 tests pass in 49 s** (`pytest dryer_optimizer/tests`).
- Default grid 48×112 → 16,288 DOF (`48·113 + 49·112 + 48·112`). Measured forward solve ≈ **188 s** at 48×112,
  ≈ 24 s at 24×56, ≈ few s at 16×32. A 30-iteration run at 48×112 ≈ 1.5–2 h → **does not meet the
  seconds-to-minutes target** (bottleneck analysis in §Performance and GAP doc).
- **No DXF export** anywhere in production (SVG + STEP exist).
- **No LU factorization reuse**: the adjoint does a fresh `spsolve(A.transpose())`.
- Conventions: SI units (m, kg, s, Pa). Domain = Y-Z side section, X = airflow depth, Y = height.
  Rows are indexed from the bottom (row 0 = bottom wall). u on vertical faces `(ny, nx+1)`, v on
  horizontal faces `(ny+1, nx)`, p at cell centers `(ny, nx)`.

Reference shorthand: TF = topflow · STO = Stokes_TO · CA = cashocs · TY = topy · DA = dolfin-adjoint ·
SOF = shapeOptimizationFoam. Line/symbol references are to `research/PIPELINE_TRACE.md`.

---

## 1. `config.py` — configuration

- **Purpose:** frozen dataclasses defining the complete problem (dryer dimensions, grid, physics, objective, optimization).
- **Classes:** `DryerConfig` (20-tray CAD-derived layout; `domain_width ≈ 0.702 m`, `chamber_height = 1.630 m`),
  `GridConfig` (nx, ny), `PhysicsConfig` (fluid + fan + nonlinear solver), `ObjectiveConfig` (weights, bands, pressure cap),
  `OptimizationConfig` (steps, filter radius, projection β), `AppConfig` (aggregate, `with_overrides`, YAML load).
- **Key values:** `alpha_min=0.0`, `alpha_max=1e5`, `alpha_interpolation_q=0.05`, `tray_resistance=0.25`,
  `fan_source_direction=-1.0`, fan curve `(0,1800)…(1.0,0)` Pa, `nonlinear_max_iterations=120`,
  `nonlinear_tolerance=1e-8`, `nonlinear_relaxation=0.3`, `velocity_reference=0.10`,
  `maximum_pressure_drop=300`, `minimum/maximum_solid_fraction=0.05/0.15`, `volume_constraint_weight=20`,
  `pressure_constraint_weight=20`, `step_size=0.08`, `move_limit=0.04`, `filter_radius_cells=2`,
  `projection_beta=1→8` ramp every 10, `binary_threshold=0.5`.
- **Inputs/outputs:** nothing computed; consumed by every module. YAML loader validates unknown sections.
- **Dependencies:** dataclasses, yaml.
- **Algorithm:** pure configuration + validation.
- **Complexity:** O(1).
- **Numerical risks:** none.
- **Correctness risks:** low. `alpha_interpolation_q=0.05` is a very low penalty exponent (near-linear
  interpolation); combined with `alpha_max=1e5` the intermediate-density resistance is mild — see GAP §BP.
- **Performance risks:** none.
- **Relevant references:** TF `topFlow.m` parameter block (continuation `qavec`, `alphamax`, volfrac);
  CA `demo_pipe_bend.py` (`alpha_in/out`, λ volume weight); STO `FluidTopoOptDriver.cpp`.
- **Recommended changes:** none required. Add `continuation_q_stages` and `volume_fraction_target` fields
  when the OC/continuation upgrades land.

## 2. `geometry/domain.py` — grid and masks

- **Purpose:** structured Y-Z grid and all boolean masks that define where design is allowed.
- **Classes/functions:** `Grid` (dx, dy, cell_x/cell_y, u/v/p shapes, `nearest_cell_row`),
  `DryerGeometry` (masks + fan/tray bookkeeping), `build_dryer_geometry(...)`.
- **Mathematical formulation:** cell-centered design density ρ over `p_shape`; MAC face arrays for u/v.
  Masks: `fixed_solid` (walls, tray floors, partition), `fan_mask` (315 mm plug-fan zone in the 350 mm
  mechanical room), `tray_forbidden` (rows around each tray floor), `boundary_clearance`,
  `outlet_clearance` (rightmost columns), `design_mask = ~(all of the above) & outside tray columns`.
  Design region = rear supply plenum + mechanical room + front return plenum.
- **Inputs/outputs:** `DryerConfig+GridConfig → DryerGeometry` (all masks validated: 20 tray masks,
  `forbidden_mask == ~design_mask`, no design∩fixed_solid, no design∩fan).
- **Dependencies:** numpy.
- **Algorithm:** analytic rectangle rasterization of the CAD-derived layout.
- **Complexity:** O(ny·nx).
- **Numerical risks:** low.
- **Correctness risks:** **medium.** The design region excludes the tray chamber entirely (baffles cannot
  be placed between trays — only in plenums/mechanical room). This matches the fixed-trays requirement,
  but note the *flow-routing freedom* is therefore restricted to three boxes. The `nearest_cell_row`
  rounding must stay consistent with `tray_elevations` (currently consistent).
- **Performance risks:** none (once per run).
- **Relevant references:** STO `FluidTopoOptDriver.cpp::Init_Boundary` (`cell_fixed` field concept);
  TF `problems.m` (domain/BC definition); CA `regular_mesh`.
- **Recommended changes:** none structurally. Optionally expose the masks in `topology.npz` for
  downstream verification.

## 3. `geometry/trays.py` — tray sampling

- **Purpose:** tray-level diagnostics; defines the sampled tray velocity used by the objective.
- **Functions:** `summarize_trays`, `tray_velocity_samples(geometry, u)`.
- **Mathematical formulation:** for each tray, sample row = floor row + 1 (the air gap just above the tray
  floor); `u_tray,r = mean(u[row, tray_col_min:tray_col_max])` — the **horizontal** velocity across the tray.
- **Inputs/outputs:** `(geometry, u) → np.ndarray(20,)`.
- **Dependencies:** numpy.
- **Algorithm:** vectorized mean over columns.
- **Complexity:** O(ny·nx) worst, effectively O(20·ncols).
- **Numerical risks:** none.
- **Correctness risks:** **medium.** Sampling only the horizontal component u in the gap row ignores the
  vertical velocity component; for a Y-Z section where air crosses trays horizontally this is a defensible
  engineering proxy, but it is a *proxy* — document it and cross-check against 3D CFD. The gap row is 1
  cell above the floor; with dy ≈ 14.6 mm this is within the 25 mm air gap (fine).
- **Performance risks:** none.
- **Relevant references:** multi-tray dryer papers (uniformity index); CA pipe-bend objective uses
  full-field integrals instead — our tray-line sampling is the project-specific choice.
- **Recommended changes:** keep; add a unit test of the sampling indices.

## 4. `geometry/export.py` — contour + file export

- **Purpose:** extract 0.5-contours from the binary topology and write NPZ/SVG deliverables.
- **Functions:** `extract_contours(binary)` (scikit-image `find_contours` with a cell-edge-loop fallback),
  `save_topology_npz(...)`, `save_topology_svg(...)` (mm-scaled SVG paths).
- **Mathematical formulation:** marching-squares contour at level 0.5 of the binary field; SVG paths in mm.
- **Inputs/outputs:** binary (ny,nx) → contour polylines (pixel coords, row-from-top convention).
- **Dependencies:** numpy, scikit-image (optional; fallback is pure numpy).
- **Algorithm:** standard contouring; **no cleanup, no min-feature filtering, no disconnected-island removal.**
- **Complexity:** O(ny·nx).
- **Numerical risks:** low.
- **Correctness risks:** **medium.** `find_contours` returns row-from-top coordinates; `save_topology_svg`
  flips to `(ny − row)` — verified by inspection. There is **no test** of contour correctness or of the
  SVG output. No DXF.
- **Performance risks:** none.
- **Relevant references:** TF `export.m` (0.5-contour → DXF LINE entities — the missing DXF stage);
  TY `visualisation.py` (STL).
- **Recommended changes:** (1) add DXF export following TF `export.m`; (2) add geometry cleanup
  (min-feature, connected-component filter) before export; (3) add contour round-trip tests.

## 5. `physics/brinkman.py` — porosity interpolation

- **Purpose:** design density → inverse-permeability α and its derivative.
- **Functions:** `resistance_from_density(ρ; α_min, α_max, penalty)`; `resistance_derivative(...)`;
  `mesh_scaled_alpha_max(μ, dx, dy, factor=1000)`.
- **Mathematical formulation (Borrvall–Petersson rational, ρ = solid fraction):**
  `α(ρ) = α_min + (α_max − α_min)·q·ρ/(q + 1 − ρ)`,
  `α′(ρ) = (α_max − α_min)·q·(q+1)/(q+1−ρ)²`.
  ρ=0 → fluid; ρ=1 → solid. (TF uses the equivalent family with the reciprocal penalty and opposite
  density convention: `α = α_min + (α_max−α_min)(1−x)/(1+q·x)`, x = fluid fraction.)
- **Inputs/outputs:** density (ny,nx) → α, α′.
- **Dependencies:** numpy.
- **Algorithm:** vectorized.
- **Complexity:** O(ny·nx).
- **Numerical risks:** low. `α_max=1e5` ≫ the `2.5μ/h²` heuristic used by TF/CA (≈ 2.6 at default cells)
  and ≫ `mesh_scaled_alpha_max` (≈ 1.5e3) → strong penalization, which is good for leakage suppression.
- **Correctness risks:** low — formula matches the field standard (TF `topFlow.m`; STO `FluidTopoOpt.cpp::alpha`).
- **Performance risks:** none.
- **Relevant references:** TF `topFlow.m` MATERIAL INTERPOLATION; STO `FluidTopoOpt.cpp:108-122`; CA demo.
- **Recommended changes:** none for the formula. Continuation on `q` is the missing regularization feature
  (GAP §continuation).

## 6. `physics/fan.py` — fan curve

- **Purpose:** piecewise-linear static-pressure-rise curve ΔP(Q).
- **Classes:** `FanCurve` (`from_pairs`, `pressure_and_slope`, `pressure`, `slope`, `maximum_flow`,
  `shutoff_pressure`).
- **Mathematical formulation:** monotone-non-increasing piecewise-linear ΔP(Q), clamped at both ends
  (slope 0 beyond the last point; first-segment slope for Q<0 but pressure clamped at shutoff).
- **Inputs/outputs:** Q (m³/s) → (ΔP Pa, dΔP/dQ Pa·s/m³).
- **Dependencies:** numpy.
- **Algorithm:** `searchsorted` segment lookup; exact segment slope.
- **Complexity:** O(log n_segments).
- **Numerical risks:** low. Clamping at Q<0 gives dP/dQ = first-segment slope (nonzero) — this enters the
  Jacobian; at startup Q≈0 the slope is steep (≈ −600 Pa/(m³/s)), which is *physically the stall slope*
  and is exactly what made early solves delicate (already mitigated by warm starts).
- **Correctness risks:** low; tested (interpolation + slope + clamping).
- **Performance risks:** none.
- **Relevant references:** fan-curve-driven internal-fan modeling from the multi-tray dryer papers
  (Song et al. 2016 analog).
- **Recommended changes:** none.

## 7. `physics/boundary_conditions.py` — BCs

- **Purpose:** Dirichlet data for the sealed-box model.
- **Classes/functions:** `BoundaryConditions` (u_fixed/u_values, v_fixed/v_values, p_fixed/p_values),
  `make_boundary_conditions(geometry, physics)`.
- **Mathematical formulation:** all outer u faces (row 0, ny−1; col 0, nx) and all outer v faces fixed to 0
  (no-slip, sealed). Four corner p cells fixed at 0 (structural singularity fix) and one mechanical-room
  fluid cell `p_fixed[ny−ny//8, nx//2]` anchors the pressure null space.
- **Inputs/outputs:** geometry → per-face/per-cell Dirichlet arrays.
- **Dependencies:** numpy.
- **Algorithm:** direct construction.
- **Complexity:** O(ny·nx).
- **Numerical risks:** **medium.** The solver's momentum equations use unusual boundary pressure entries
  (`+1/dx` at col 0, `−1/dx` at col nx for u; `+2/dy` at row 0, `−2/dy` at row ny for v) — these are
  one-sided/halo-less stencils whose correctness is asserted only by the residual test, **not** by an
  analytic solution. The corner p-fix trick and the 2/dy entries need an analytic-flow verification.
- **Correctness risks:** **medium** (sealed model is intentional and tested for opening flows = 0; but
  boundary stencil correctness is unverified analytically).
- **Performance risks:** none.
- **Relevant references:** TF `problems.m` (inlet/outlet Dirichlet + outlet pressure pin); SOF `adjointBC.H`
  (adjoint BC treatment); STO `Init_Boundary`.
- **Recommended changes:** add an analytic channel-flow test that exercises these stencils (GAP §verification).

## 8. `physics/solver.py` — the forward solver (526 lines, core)

- **Purpose:** assemble and solve the steady incompressible fan-coupled MAC system; returns the converged
  state **and the final analytic Jacobian** used by the adjoint.
- **Classes/functions:** `FlowState` (solution, u/v/p fields, matrix, rhs, fan_flow/pressure/slope,
  opening flows, nonlinear diagnostics), `BrinkmanSolver` (`assemble`, `_residual`, `solve`, `residual_norm`,
  `fan_flow_and_curve`, `_u_alpha/_v_alpha`, `_u_convection/_v_convection`, index maps).
- **Mathematical formulation** (steady NS–Brinkman, upwind convection, effective viscosity μ = μ_mol + μ_eddy):
  - DOF vector x = [u (ny·(nx+1)); v ((ny+1)·nx); p (ny·nx)].
  - u-momentum at face (r,c): `(α_u + κ)u − μ/dx²(u_{c±1}) − μ/dy²(u_{r±1}) + (p_c − p_{c−1})/dx = f_fan + conv_u`,
    where κ = μ·(2/dx² + 2/dy²) interior (boundary-reduced at walls); α_u = arithmetic face mean of α
    (boundary value at edge faces).
  - v-momentum analogous with `±1/dy` (and `±2/dy` at top/bottom) pressure coupling.
  - Continuity: `(u_{c+1}−u_c)/dx + (v_{r+1}−v_r)/dy = 0`; Dirichlet p cells become identity rows.
  - Fan: `Q = dir·b·dy·Σ u[fan_rows, fan_face_col]`; `ΔP = curve(Q)`; body force `f_fan = dir·ΔP/L_fan`
    applied on `fan_source_cols × fan_rows`. The full Jacobian adds the low-rank fan coupling
    `dΔP/dQ·(b·dy)` across all fan-row momentum equations (dense fan-row block) and the exact upwind
    convection Jacobian (nonsymmetric).
  - Newton: assemble residual via Oseen matrix `A(x)x − b`; assemble full Jacobian `J`; solve
    `J·δ = Jx − r`; damped update `x += λ·δ`, λ from 0.3 halving ×10 backtracking on ‖r‖∞;
    convergence ‖r‖∞<1e-8 **and** ‖Δx‖∞<1e-8; max 120; warm start from previous converged state.
- **Inputs/outputs:** `(density, initial_solution?) → FlowState` (includes matrix/rhs for adjoint reuse).
- **Dependencies:** scipy.sparse (csr, spsolve), numpy.
- **Algorithm:** per-cell Python-loop assembly (lists → `csr_matrix`), SuperLU `spsolve` per Newton step.
- **Complexity:** assembly O(DOF) with a large Python constant (per-cell loops + dict-based convection
  derivatives); each Newton iteration does ≥2 full assemblies (residual pass + Jacobian pass) plus up to 10
  backtracking residual assemblies → **≈ 12 assemblies per Newton iteration**; measured 188 s/solve at
  48×112. Factorization ~O(DOF^1.5) per linear solve and is a small fraction of wall time at 16k DOF.
- **Numerical risks:** **high (convergence).** The piecewise-linear fan curve creates a steep dP/dQ at
  startup (stall slope); backtracking is load-bearing. The `_residual` re-derives the Oseen matrix rather
  than reusing `A·x−b` from the Jacobian assembly — correctness of the "converged Jacobian ≠ residual
  matrix" split is subtle and only tested via the residual-norm contract.
- **Correctness risks:** **medium-high.** Convection + eddy viscosity ⇒ effective Re ≈ 165; the solver is
  *not* the Stokes system of the reference pipeline, and no analytic/benchmark flow validates the
  upwind stencils. The final Jacobian includes `dΔP/dQ` and convection derivatives (good for the adjoint),
  but the stored `rhs` is reconstructed as `matrix·x − residual` so that `matrix·x − rhs` reproduces the
  residual — an invariant that deserves a dedicated test (currently only `residual_norm` is tested).
- **Performance risks:** **highest in the codebase** — see §Performance.
- **Relevant references:** STO `FluidTopoOpt.cpp::Update_Fluid/Update_Grad` (MAC-grid discrete-adjoint
  pattern); TF `topFlow.m` Newton loop (`nltol=1e-6`, retry-from-zero); DA `blocks/solving.py`
  (adjoint of the *same* operator).
- **Recommended changes:** (1) vectorized assembly (single numpy pass producing both Oseen and full
  Jacobian) — biggest speedup; (2) return the LU factorization for the adjoint (splu reuse);
  (3) reduce backtracking residual assemblies (reuse `A·x−b` from the last Jacobian assembly);
  (4) add Stokes-mode switch (convection off) as the production baseline for optimization robustness.

## 9. `optimization/filters.py` — filter + projection

- **Purpose:** density filter (length-scale) and smooth projection (near-0/1).
- **Functions:** `DensityFilter` (uniform-weight symmetric neighborhood, exact `transpose_apply`),
  `smooth_projection` (tanh).
- **Mathematical formulation:** `ρ_f = Wρ` with `W` the normalized convolution over a square radius-r
  neighborhood (uniform weights); projection
  `ρ_p = [tanh(βη) + tanh(β(ρ_f−η))]/[tanh(βη)+tanh(β(1−η))]`, derivative included.
- **Inputs/outputs:** (ny,nx) arrays; filter matrix (ny·nx)² sparse.
- **Dependencies:** scipy.sparse, numpy.
- **Algorithm:** sparse-matrix multiply.
- **Complexity:** filter build O(ny·nx·r²); apply O(nnz) ≈ O(ny·nx·r²).
- **Numerical risks:** low. Uniform weights over a square stencil are the simplest choice; the
  symmetric-transpose property is tested (`test_filter_transpose_identity`).
- **Correctness risks:** low. (TY's `filter_sens_sigmund` is the older heuristic; our chain-rule-transpose
  filter is the consistent choice.)
- **Performance risks:** low (sparse matmul).
- **Relevant references:** TY `topology.py:317 filter_sens_sigmund` (heuristic contrast); 88-line chain-rule
  filter; CA projection practice.
- **Recommended changes:** none; keep.

## 10. `optimization/objective.py` — the engineering objective

- **Purpose:** airflow-uniformity objective with pressure cap and volume band.
- **Classes/functions:** `ObjectiveValue`; `_target_velocity`; `evaluate_objective`;
  `objective_state_gradient` (required kwarg `fan_source_direction`); `objective_density_gradient`.
- **Mathematical formulation** (exact):
  - Tray averages `u_r = mean(u[sample_row_r, tray_col_min:tray_col_max])`, r = 1..20.
  - Target `v_t = Q/(N·b·span)`, `span = max(tray_depth, dx)`, Q signed (positive = correct direction).
  - Uniformity error `J_uni = (1/N)Σ_r (u_r − v_t)² / v_ref²`.
  - CV = σ(u_r)/max(|mean|, 1e-12).
  - Pressure penalty `J_p = w_p·(max(0, ΔP − ΔP_max)/ΔP_ref)²`, ΔP = fan static pressure.
  - Volume band penalty `J_vol = w_vol·(max(0,f−f_max)² + max(0,f_min−f)²)`, f = mean ρ over design cells.
  - `J = w_u·J_uni + w_m·f + J_vol + J_p`  (`w_u=1, w_m=0.01, w_vol=20, w_p=20`).
  - State gradient: tray rows → `dJ/du` from deviation; fan faces → chain rule
    `dJ/dQ·dQ/du`, `dQ/du = dir·b·dy` (target term) and `dJ/dΔP·dΔP/dQ·dQ/du` (pressure term).
  - Explicit density gradient: material cost + volume-band derivative over design cells.
- **Inputs/outputs:** `(state, geometry, config) → ObjectiveValue` and gradients.
- **Dependencies:** numpy.
- **Algorithm:** vectorized numpy + per-tray loops.
- **Complexity:** O(ny·nx).
- **Numerical risks:** **high (engineering).** Because `v_t ∝ Q`, the *target moves with the flow*: at
  constant tray distribution, reducing Q reduces J_uni. Combined with the pressure cap (which penalizes
  high ΔP, i.e. high Q on a falling curve), the optimizer is structurally tempted to *turn the fan down*
  instead of homogenizing — the "everything is uniform because everything is stagnant" degeneracy. The
  pressure cap's `max(0,·)` makes J_p piecewise-smooth (fine for adjoint) but the degeneracy must be
  verified empirically (see GAP §objective).
- **Correctness risks:** **medium.** The state gradient is analytic but FD-verified only for the reduced
  objective (material/volume/pressure weights zeroed, coarse grid, 3 cells, single ε) — the **active
  pressure-cap term and the volume-band term are never FD-verified**.
- **Performance risks:** none.
- **Relevant references:** CA `demo_pipe_bend.py` (dissipation objective `∫(μ|∇u|²+α|u|²)` — the
  pressure-drop surrogate); TF `analyticalElement.m` (objective integrand); multi-tray papers (uniformity index).
- **Recommended changes:** (1) full-objective FD verification including an active pressure cap;
  (2) evaluate the Q-degeneracy: options are a *fixed* target velocity or a CV-only term; (3) document the
  tray-sampling proxy.

## 11. `optimization/adjoint.py` — discrete adjoint

- **Purpose:** solve `Aᵀλ = ∂J/∂x` and form `dJ/dρ`.
- **Functions:** `AdjointResult`; `compute_adjoint`; `_add_face_contribution`.
- **Mathematical formulation:** λ = A⁻ᵀ·g with g = `objective_state_gradient(...)`
  (`fan_source_direction` forwarded from `solver.physics` — the fixed plumbing from the earlier bug hunt).
  `dJ/dρ = g_ρ − Σ_faces λ_f·u_f·(∂α/∂ρ contribution)`, where each interior u-face splits ½·α′ to the two
  adjacent cells, v-faces split ½ (interior) or 1 (boundary), zeroed outside `design_mask`.
- **Inputs/outputs:** `(solver, state, geometry, objective, config) → AdjointResult`.
- **Dependencies:** scipy.sparse (`spsolve` on transpose), numpy.
- **Algorithm:** one sparse solve of the transposed converged Jacobian; per-cell loops over faces.
- **Complexity:** spsolve O(DOF^1.5) (fresh factorization, no reuse); face loops O(ny·nx) Python.
- **Numerical risks:** **medium.** A fresh `spsolve(A.transpose())` refactors; SuperLU on a transpose is
  correct but wasteful — the factorization is available from the forward solve. Non-finite guard exists.
- **Correctness risks:** **medium.** The FD test (3 cells, ε=1e-5, 12×24, reduced objective, tol 2e-4) is a
  good smoke test but not a full verification: not multi-ε, not the full objective, not the active
  pressure cap. The density-gradient face-splitting logic is only exercised at the coarse grid.
- **Performance risks:** one extra factorization per iteration (~30–50% of the linear-solve budget).
- **Relevant references:** TF `topFlow.m` ADJOINT SOLVER (`L = J'\RHS`, transpose of the *same* Jacobian);
  STO `FluidTopoOpt.cpp::Update_Grad` (adjoint RHS → same solver); DA `GenericSolveBlock`.
- **Recommended changes:** (1) return/accept the LU factorization from the forward solve and reuse it
  (`splu(A).solve(Aᵀ λ = g)` via the transposed factor or `lu.solve` with `trans='T'`);
  (2) extend the FD test to the full objective, multiple ε, several design cells.

## 12. `optimization/update.py` — density update

- **Purpose:** raw-design update loop step (V1).
- **Functions:** `physical_density` (filter → projection → forbidden zeroing), `update_density`
  (chain-rule backprop of the physical gradient through projection and filter transpose, then bounded
  move), `initial_density` (uniform f_init over design mask), `OptimizationHistory`.
- **Mathematical formulation:** `ρ_raw ← clip(ρ_raw − α·ĝ, ρ_raw ± m, [0,1])` with
  `ĝ = normalize(filterᵀ(proj′·g_phys))`; α = step_size 0.08, m = move_limit 0.04. **This is normalized
  gradient descent, not OC.**
- **Inputs/outputs:** raw density + physical gradient → updated raw density.
- **Dependencies:** numpy.
- **Algorithm:** vectorized numpy + sparse filter transpose.
- **Complexity:** O(ny·nx).
- **Numerical risks:** low.
- **Correctness risks:** **medium.** There is no volume constraint in the OC sense — only the soft band
  penalty inside the objective; the update may violate the [0.05, 0.15] band. Move limit (0.04) vs
  step (0.08) means the clip often binds — the effective step is ~move limit.
- **Performance risks:** none.
- **Relevant references:** TF `topFlow.m` OC update (bisection on λ); TY `update_desvars_oc` (OC in pure
  Python); STO `OptimizerMma.h` (MMA when multiple constraints).
- **Recommended changes:** replace with an **OC update with Lagrange-multiplier bisection** on the volume
  fraction (GAP §OC) once correctness of the gradient is established.

## 13. `visualization/plots.py` — diagnostics plots

- **Purpose:** matplotlib PNG diagnostics (density, speed, pressure; distribution bars; history curves).
- **Functions:** `plot_flow`, `plot_optimized_distribution` (CAD-bridge plot), `plot_history`.
- **Mathematical formulation:** speed = cell-averaged face velocities; tray CV re-derived for the bar plot.
- **Inputs/outputs:** state/geometry/objective/history → PNG.
- **Dependencies:** matplotlib (imported lazily — headless-safe).
- **Algorithm:** imshow/contour/bar.
- **Complexity:** O(ny·nx).
- **Numerical risks:** none.
- **Correctness risks:** low (diagnostics only).
- **Performance risks:** low (once per run).
- **Relevant references:** TF `postproc.m`; TY `visualisation.py`.
- **Recommended changes:** none required.

## 14. `main.py` — optimization driver

- **Purpose:** end-to-end loop: physical density → solve (warm start) → objective → adjoint → update →
  β-ramp → save NPZ/SVG/JSON/PNG.
- **Functions:** `run_optimization`, `_save_outputs`, CLI (`main`, `_parse_args`).
- **Mathematical formulation:** orchestration only. Projection β doubles every 10 iterations up to 8;
  warm start uses the previous converged state (avoids fan shutdown restart).
- **Inputs/outputs:** config → `OptimizationResult` + `data/output/{topology.npz, topology.svg,
  summary.json, flow.png, history.png}`.
- **Dependencies:** numpy, tqdm, yaml.
- **Algorithm:** sequential loop.
- **Complexity:** O(iterations × solve cost).
- **Numerical risks:** none beyond the solver.
- **Correctness risks:** low.
- **Performance risks:** inherits the solver; no convergence-based early stop (fixed `iterations` count).
- **Relevant references:** TF `topFlow.m` continuation loop (early stop on change < 1e-3 × 5).
- **Recommended changes:** add change-based early stopping and per-stage timing instrumentation.

## 15. `sldw_optimized.py` — CAD bridge (564 lines, non-core)

- **Purpose:** extrude the 2D binary topology to 3D baffles and build the full dryer assembly; export STEP.
- **Classes/functions:** `CadDimensions` (mm contract: 702 × 1630 mm), `load_topology`, `_contiguous_runs`,
  `_build_baffles` (per-row runs → Box extrudes), `_build_trays`, `_build_dryer_assembly`
  (enclosure, false wall, diffuser, supply wedge, turning cowl + vanes, dummy fan, partition, lids),
  `build_and_export`, CLI.
- **Mathematical formulation:** run-based box extrusion of the binary mask; Y-Z → CAD Y-Z mapping.
- **Inputs/outputs:** `topology.npz` → `sldw_dump/dryer_optimized_assembly.step`,
  `optimized_baffles.step`, `cad_manifest.json`, distribution PNG.
- **Dependencies:** build123d (imported lazily — not in `requirements.txt`; installed in the venv).
- **Algorithm:** constructive solid geometry.
- **Complexity:** O(binary cells).
- **Numerical risks:** low.
- **Correctness risks:** **medium.** Baffle solids are unions of per-run boxes; overlapping/grazing cells
  may create non-manifold or zero-thickness features — no validation of the STEP solid (manifold check).
- **Performance risks:** low.
- **Relevant references:** TY STL export; TF `export.m` DXF (2D before extrusion).
- **Recommended changes:** later — add manifold/watertight checks and optionally DXF for 2D fabrication.

## 16. `tests/` — coverage matrix (11 tests, 49 s)

| File | Tests | What is actually verified |
|---|---|---|
| `test_flow.py` | 4 | CAD constraints + 20 trays; fan-curve interpolation/slope/clamp; forward solve finite + residual < 1e-7 + fan operating point on curve; sealed openings = 0; pressure anchors p=0 |
| `test_adjoint.py` | 2 | **FD gradient** (3 design cells, ε=1e-5, reduced objective, coarse grid, tol 2e-4); **fan_source_direction sign-flip plumbing** + adjoint state-gradient identity |
| `test_optimization.py` | 4 | filter transpose identity; density update bounds/mask/move-limit; npz round-trip; 2-iteration end-to-end |
| `test_cad_bridge.py` | 2 | dimension contract (702×1630 mm); contiguous-run extraction |

**Missing tests (must be added — detailed in the GAP doc):** mass conservation; analytic flow
(Poiseuille/obstacle); no-slip boundary check; full-objective multi-ε FD gradient incl. active pressure
cap; volume-constraint behavior; contour/SVG/DXF output; geometry cleanup; performance regression.

## 17. Cross-cutting: dependencies

`requirements.txt`: numpy, scipy, matplotlib, pandas, pyyaml, tqdm, scikit-image. **pandas is not used by
any audited module** (remove or justify). build123d is used by `sldw_optimized.py` but not listed.
Everything else is used. This stays a lightweight stack — no FEniCS/OpenFOAM/CUDA — matching the
production constraint.

---

## Summary: what the codebase is

A **dimensional (SI), 2D MAC-grid, fan-coupled steady NS–Brinkman solver with upwind convection, discrete
adjoint (transposed converged Jacobian), explicit density filter + tanh projection, and normalized
gradient-descent updates with penalty-based volume/pressure control**. The math that matters (Brinkman
interpolation, discrete adjoint, fan-curve coupling, filter transpose, projection) is the correct,
reference-consistent family. The three dominant gaps are (1) verification depth (adjoint FD is a smoke
test, no analytic flow cases, no mass-conservation test), (2) an objective whose target velocity is
proportional to the flow the optimizer controls (degeneracy risk), and (3) a Python-loop assembly +
no-factorization-reuse performance profile (~hours per 48×112 run instead of minutes). Details and
stage-by-stage classification: `IMPLEMENTATION_GAP_ANALYSIS.md`.
