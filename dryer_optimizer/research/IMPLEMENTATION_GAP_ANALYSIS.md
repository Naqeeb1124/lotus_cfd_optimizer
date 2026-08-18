# IMPLEMENTATION_GAP_ANALYSIS

Compare the existing production implementation (audited in `CURRENT_IMPLEMENTATION_AUDIT.md`) against the
target architecture and the engineering objective:

> Optimize the internal 2D flow-routing geometry of a multi-tray dryer for maximum airflow uniformity
> across the trays, subject to fixed outer geometry, fixed trays/forbidden regions, manufacturable
> geometry, bounded design volume, minimum feature/channel size, acceptable pressure drop, and physically
> meaningful airflow — ending in a real 2D DXF/manufacturable geometry.

Target pipeline:
`MAC grid → Stokes/Brinkman → Borrvall–Petersson → forward solve → uniformity objective → discrete adjoint
→ sensitivity → OC update → volume constraint → continuation → filtering/projection → contour → DXF`.

Classification legend: **[COMPLETE]** verified · **[PARTIAL]** works but incomplete/unverified ·
**[MISSING]** absent · **[INCORRECT]** wrong as-is · **[NEEDS_VERIFICATION]** implemented but not trusted.

---

## Stage-by-stage classification

| # | Stage | Class | Evidence / notes |
|---|-------|-------|------------------|
| 1 | 2D staggered MAC grid | [COMPLETE] | `Grid`/`DryerGeometry` (`geometry/domain.py`): p (ny,nx) centers, u (ny,nx+1), v (ny+1,nx) faces; validated masks. Matches STO `mac_grid` pattern. |
| 2 | Stokes/Brinkman flow | [NEEDS_VERIFICATION] | Solver is **steady NS–Brinkman with upwind convection** (`physics/solver.py`), effective Re ≈ 165. All flow references (TF/STO/CA) are Stokes; the convective terms are an unverified extension (1st-order upwind ⇒ numerical diffusion). **No analytic-flow test exists.** The Stokes limit (`convection_enabled=False`) is available but is not the default. |
| 3 | Borrvall–Petersson interpolation | [COMPLETE] | `resistance_from_density` matches the rational law and its derivative; consistent with TF `topFlow.m` / STO `alpha()`. `α_max=1e5` ≫ heuristic `2.5μ/h²` (safe). |
| 4 | Forward solve | [COMPLETE] (perf-limited) | Damped Newton on the exact residual; final Jacobian stored for the adjoint; warm starts; fan operating point on curve (tested). Correctness of boundary stencils unverified analytically; cost ~188 s/solve at 48×112 (see §Performance). |
| 5 | Airflow-uniformity objective | [PARTIAL] | Exact formulas present and self-consistent (`optimization/objective.py`), but (a) `v_t ∝ Q` **target-velocity degeneracy** — the optimizer can trivially lower J by reducing fan flow; (b) tray sampling is a horizontal-velocity proxy; (c) no unit test of the objective value itself. |
| 6 | Discrete adjoint | [COMPLETE] (under-verified) | `compute_adjoint` solves `Aᵀλ = ∂J/∂x` with the **converged forward Jacobian** — the reference-correct pattern (TF `J'\RHS`, DA). FD test exists but is a smoke test (see §Verification). |
| 7 | Sensitivity / chain rule | [COMPLETE] | State gradient (incl. fan `dQ/du = dir·b·dy` target + pressure terms) and density gradient (face splitting ½/1, mask zeroing) are analytic and structurally correct; `fan_source_direction` plumbing pinned by test. |
| 8 | OC update | [MISSING] | `update_density` is **normalized gradient descent** (step 0.08, move 0.04) — the OC/Lagrange-multiplier machinery from TF/TY is not implemented. |
| 9 | Volume constraint | [PARTIAL] | Only a soft **band penalty** [0.05, 0.15] in the objective (w=20). Not an exact constraint; no bisection; no volume-fraction target field. |
| 10 | Continuation | [PARTIAL] | Projection β-continuation exists (×2 every 10 it to 8). **Brinkman-penalty continuation on q is missing** (q fixed at 0.05), and there is no Re/continuation ladder as in TF (`qavec=[q,2q,10q,20q]`). |
| 11 | Filtering / projection | [COMPLETE] | Chain-rule-transpose filter (tested) + tanh projection with β ramp (the consistent choice vs TY's heuristic filter). |
| 12 | Contour extraction | [PARTIAL] | `extract_contours` (skimage + fallback) and SVG export work, but **no cleanup** (disconnected islands, min-feature/channel checks) and **no correctness test**. |
| 13 | DXF export | [MISSING] | No DXF anywhere. SVG (mm-scaled) and build123d STEP exist; TF `export.m` gives the exact DXF recipe (0.5-contour → LINE entities). |
| 14 | Geometry ↔ production contract | [PARTIAL] | CAD dimension contract tested (702×1630 mm, STEP export). Baffle solids are unvalidated box unions (manifold/watertight risk). |
| 15 | Performance target (seconds–minutes) | [PARTIAL] | **Fails today** (≈1.5–2 h for 30×48×112). All fixes identified in §Performance; correctness must be locked first. |

**Net: 5 COMPLETE, 3 MISSING, 4 PARTIAL, 0 INCORRECT, 2 NEEDS_VERIFICATION.** Nothing is implemented
wrong-as-such; the risk concentration is (a) unverified physics/adjoint beyond the smoke test and
(b) the objective degeneracy, with performance (c) as the constraint on practical iteration.

---

## Engineering-objective audit

- **Fixed outer geometry / fixed trays / forbidden regions:** [COMPLETE] — masks enforced and tested.
- **Manufacturable geometry:** [PARTIAL] — filter radius 2 cells gives a length scale, but no
  post-contour cleanup, no min-channel guarantee, no manifold validation of the CAD solid.
- **Bounded design volume:** [PARTIAL] — soft band only; no hard volume target.
- **Minimum feature/channel size:** [PARTIAL] — implicit via filter radius (≈ 2 cells ≈ 29 mm at 48×112),
  never checked at export.
- **Acceptable pressure drop:** [PARTIAL] — soft cap via fan static pressure (300 Pa); no hard constraint,
  and "pressure drop" is the fan ΔP at the operating point, not a duct-loss measure.
- **Physically meaningful airflow:** [NEEDS_VERIFICATION] — the sealed recirculating model is physically
  coherent and produces through-tray flow on ≥16×32 grids, but the target-velocity degeneracy can drive
  the optimizer to *reduce* flow rather than homogenize it. **This is the single most important
  engineering question to resolve.**
- **DXF/manufacturable output:** [MISSING] — see stage 13.

---

## Performance investigation (target: seconds–minutes per practical run)

Measured: 48×112 (16,288 DOF) ≈ **188 s/solve**; 24×56 ≈ 24 s; 16×32 ≈ few s. 30 iterations × ~3 min ⇒
**1.5–2 h/run today**. Per-iteration budget breakdown (est., 48×112): Newton ~10–40 iterations; each
Newton iteration = ≥2 full assemblies + up to 10 backtracking residual assemblies; assembly is a
**per-cell Python loop** with dict-based convection derivatives (≈16k equations × Python overhead).

1. **Sparse matrix assembly — THE bottleneck.** `assemble()` builds rows/cols/values with Python-level
   loops over all u/v/p cells, plus Python-dict convection derivative maps. At 48×112 this is the dominant
   wall-time term (not the factorization). **Fix:** vectorize with NumPy (precomputed stencil index arrays
   and broadcast coefficient arrays → `coo_matrix`), or build the system as a fixed sparsity-pattern matrix
   and update `data` only (density/iterate change values, not pattern) — this alone is expected to cut
   assembly cost 10–50×.
2. **Factorization reuse between primal and adjoint — NOT done.** The adjoint calls
   `spsolve(A.transpose())`, re-factorizing the transposed matrix. The forward solve already factored the
   converged Jacobian. **Fix:** `lu = splu(A)`; solve `lu.solve(b, trans='T')` for the adjoint. Removes a
   full factorization per iteration (~25–40% of the linear-solve budget). (Newton iterations must still
   refactor — the matrix changes with x; only the final converged factor is reusable.)
3. **Newton backtracking re-assembly.** Each trial residual evaluation calls `_residual` → full Oseen
   `assemble`. Up to 10 trials per iteration ⇒ up to ~12 assemblies per Newton iteration.
   **Fix:** compute the trial residual as `A_oseen·x_trial − b` from the last assembled matrices (update
   data only), and/or derive the Oseen residual from the full Jacobian assembly in a single pass (one
   assembly returns both R and J). Also cap backtracking to ~4 trials; the fan-curve nonlinearity is only
   in a small low-rank block.

**Secondary findings:** `spsolve` (SuperLU) is fine at 2D scale — no iterative solver needed below
~100k DOF; no preconditioner needed. No dense arrays of consequence (fan coupling is a small low-rank
block). Memory copies of u/v/p per iteration are minor. `physical_density`/filter/projection/objective/
gradient are vectorized and cheap. Grid refinement 48×112→96×224 raises DOF 4× and sparse factor cost
~8× — the default 48×112 is the right operating point for now. Optimization iteration count (30, no early
stop) adds no direct cost but no early-stop savings either.

**Target feasibility:** with fixes 1–3, expect ~5–20× speedup ⇒ 48×112 solve ≈ 10–40 s ⇒ 30-iteration run
≈ 5–20 min; a 32×64 design sweep lands in seconds-to-minutes. This meets the goal *after* correctness is
established (see order below).

---

## Verification / test coverage analysis

Requested 12 categories vs current 11 tests:

| # | Test | Status | Action |
|---|------|--------|--------|
| 1 | Forward solver correctness | [PARTIAL] | residual-norm + fan-curve operating point tested; add **analytic channel flow (Poiseuille) and Couette** cases |
| 2 | Mass conservation | [MISSING] | add: global Σ(face fluxes)=0 over all cells ≤ 1e-10 on solved states; per-cell divergence ≤ tol |
| 3 | Pressure/velocity BCs | [PARTIAL] | sealed openings + p-anchor tested; add **no-slip check** (all fixed faces exactly 0) and analytic pressure-gradient relation |
| 4 | Known analytical flow cases | [MISSING] | add Poiseuille profile in a straight channel with α=0: u(y)=ΔP·y(H−y)/(2μL) match to <1% ; flow-past-obstacle symmetry |
| 5 | Objective correctness | [PARTIAL] | add unit tests of `evaluate_objective` formulas (target, CV, penalties) on synthetic states |
| 6 | Gradient correctness | [PARTIAL] | FD test covers 3 cells, 1 ε, reduced objective; **extend to full objective, active pressure cap, 5+ cells, ε ∈ {1e-3,1e-4,1e-5,1e-6} with convergence trend** |
| 7 | Adjoint correctness | [COMPLETE] | FD test + sign-flip plumbing test exist (still strengthen per #6) |
| 8 | FD gradient verification | [PARTIAL] | exists but single-ε; the user requirement is **several ε** and a Richardson-convergence assertion |
| 9 | Optimization update correctness | [PARTIAL] | bounds/mask/move-limit tested; add objective-monotonicity check on a smooth toy problem |
| 10 | Volume constraint | [MISSING] | no test (feature absent) — appears with the OC upgrade; test exact volume-fraction satisfaction + gradient of the constraint |
| 11 | Geometry extraction | [PARTIAL] | npz round-trip tested; add contour round-trip (known shape → contour → area/length check) and cleanup tests |
| 12 | DXF output | [MISSING] | no DXF; add writer + parser round-trip test when implemented |

**Adjoint FD requirement (explicit):** for selected design variables compute
`[J(x+εeᵢ) − J(x−εeᵢ)]/(2ε)` over several ε and require the adjoint gradient to match within
`O(ε²)` truncation (relative error ≤ 1e-4 at ε=1e-5, improving as ε shrinks). Current test asserts a
single absolute error < 2e-4 at one ε — insufficient.

---

## Reference-hierarchy adjustment (from the audit evidence)

The proposed order was STO > TF > CA > DA > TY > SOF. The code-level evidence says:

| Component | Best reference | Why |
|---|---|---|
| Brinkman interpolation | TF `topFlow.m` (≡ STO `alpha()`) | identical rational family; cleanest presentation |
| Forward solve / Newton | TF `topFlow.m` + STO `Update_Fluid` | TF Newton+line search; STO MAC-grid layout |
| Adjoint (discrete) | **TF `topFlow.m` ADJOINT SOLVER** + DA `GenericSolveBlock` | transpose-of-same-Jacobian pattern; DA as validation oracle |
| FD gradient validation | **STO `Numerical_Derivative`** | ships an analytic-vs-FD per-cell checker |
| Objective (pressure/dissipation) | **CA pipe-bend** | closest published pressure-drop NS objective |
| OC update + continuation | TF `topFlow.m`; TY `update_desvars_oc` | bisection on λ; p/q continuation |
| Filter | TY (heuristic) vs 88-line chain-rule (ours) | ours is the consistent choice; TY shows Python implementation |
| DXF export | **TF `export.m`** | exact 0.5-contour→DXF recipe |
| MMA (multi-constraint future) | STO `OptimizerMma.h` | reference for the upgrade path |
| SOF | none for production | continuous-adjoint shape optimization; least relevant |

**Verdict:** the audit *reverses part of the proposed ranking* — topflow is the primary reference for most
stages (adjoint, OC, interpolation, DXF), Stokes_TO ranks first only for the FD checker and the MMA
upgrade path. cashocs rises for the pressure-drop objective. This is documented per-component above.

---

## Required summary

**What the existing solver already gets right**
- The correct reference family end-to-end: MAC grid, Borrvall–Petersson interpolation (exact law +
  derivative), discrete adjoint on the **converged full Jacobian** (incl. fan dP/dQ coupling), transposed
  chain-rule filter + tanh projection, warm-started Newton with backtracking, and the fixed
  `fan_source_direction` plumbing.
- Sealed-box BCs with a guaranteed-fluid pressure anchor; fan operating point solved exactly on the curve.
- Clean module separation (config/geometry/physics/optimization/visualization), lazy heavy imports,
  lightweight dependency stack (no FEniCS/OpenFOAM/CUDA).
- A working 11-test suite incl. one FD adjoint smoke test and the filter-transpose identity.

**The 3 most serious correctness risks**
1. **The target-velocity degeneracy** — `v_t = Q/(N·b·span)` makes the uniformity target proportional to
   the flow the optimizer controls; the optimizer is structurally rewarded for *reducing* Q (uniformity
   trivially improves, pressure cap rewards lower ΔP). Must be resolved (fixed target, CV-only objective,
   or min-velocity constraint) before results are meaningful.
2. **Adjoint gradient unverified for the real objective** — the FD test zeroes material/volume/pressure
   weights and checks 3 cells at one ε. The active pressure-cap term, the volume-band term, and the
   fan-target chain rule are never FD-checked; the fan-pressure-slope term (`dJ_p/dΔP·dΔP/dQ`) is the
   least-tested link.
3. **Unverified physics** — upwind convection at effective Re ≈ 165 (all references are Stokes), unusual
   boundary pressure stencils (±2/dy, ±1/dx), and a reconstructed `rhs = J·x − r` invariant are asserted
   only by a residual-norm test; there is no analytic flow case, no mass-conservation check, and no
   no-slip check.

**The 3 biggest performance bottlenecks**
1. Per-cell Python assembly loops (incl. dict-based convection derivatives) — dominant wall time at 48×112.
2. No factorization reuse: adjoint re-factorizes `Aᵀ` every iteration.
3. Newton backtracking re-assembles the full Oseen matrix up to 10× per iteration (2-pass assembly:
   residual + Jacobian).

**What must be fixed before optimization results can be trusted**
1. Resolve the target-velocity degeneracy (engineering decision + regression test).
2. Full-objective multi-ε FD gradient verification (adjoint), incl. an active pressure cap.
3. Analytic flow + mass-conservation + no-slip tests to lock the forward solver and BC stencils.

**What can be left for later**
- DXF export and geometry cleanup (SVG/STEP already usable); OC/volume-bisection upgrade; q-continuation;
  MMA for multiple constraints; manifold validation of CAD solids; iterative/preconditioned solvers for
  very large grids; pandas dependency cleanup; profiling harness formalization.

**Recommended implementation order**
1. **Tests first (correctness):** analytic Poiseuille + mass conservation + no-slip; full-objective
   multi-ε FD gradient with active pressure cap. Fix anything they expose (esp. the degeneracy).
2. **Performance (mechanics):** vectorized single-pass assembly (fixed sparsity pattern, data-only
   updates); backtracking residual reuse; LU reuse for the adjoint (`splu(...).solve(..., trans='T')`).
   Re-measure 24×56 and 48×112.
3. **Update upgrade:** OC with Lagrange-multiplier bisection + volume-fraction target; then q-continuation.
4. **Manufacturability:** DXF export (TF `export.m` recipe) + contour cleanup/min-feature/channel checks.
5. **Scale study + validation run:** 48×112 × 30 with per-stage timers; benchmark vs the reference family
   (double-pipe case) and record in `research/`.
