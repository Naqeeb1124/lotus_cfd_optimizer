# Implementation Comparison — What to Use Where in Our Python Solver

**Scope.** Compare the six reference implementations and recommend an approach for each component of our
eventual **Python, 2D structured-grid** solver. Target architecture:

```
2D structured grid → Stokes/Brinkman flow solver → tray-airflow objective
→ adjoint → sensitivity → topology update → filtering → manufacturable geometry
→ 2D plot/DXF → extrusion → later 3D Fluent validation
```

**Constraint.** The production solver must NOT depend on the reference repositories. Everything below is
approach-level guidance; any borrowed algorithm is re-implemented in our own code (and all GPL-licensed
tools are only sources of ideas, never code).

Reference abbreviations: **TF** = topflow (MATLAB, BSD-3) · **STO** = Stokes_TO (C++/CUDA) ·
**CA** = cashocs (Python/FEniCS) · **DA** = dolfin-adjoint (Python/FEniCS) ·
**TY** = topy (Python) · **SOF** = shapeOptimizationFoam (C++/OpenFOAM).

---

## Component-by-component comparison

### 1. Stokes flow solver

| Repo | Approach | Notes |
|---|---|---|
| TF | Q1-Q0 mixed FEM, nonlinear NS, Newton iterations | Analytic element matrices; continuation from Stokes at low Re |
| STO | GPU velocity-pressure Stokes solve | Same physics, massively parallel |
| CA | FEniCS FEM (Taylor–Hood etc.) | Comfortable, but drags in FEniCS |
| DA / SOF | FEniCS / OpenFOAM FV | Not applicable to a standalone code |
| TY | — (no flow) | — |

**Recommendation:** keep our **MAC-grid finite-volume Stokes/Brinkman solver** (already correct and
validated). It is the natural fit for a structured grid and matches our existing code. Use TF as the
**numerical benchmark** (same physics, different discretization) for validation cases. Add **Reynolds
continuation** (TF's pattern) when we move from Stokes toward NS effects.

### 2. Brinkman penalization

| Repo | Approach |
|---|---|
| TF | `α(ρ) = α_min + (α_max − α_min)·q·(1−ρ)/(q+ρ)` (Borrvall–Petersson rational) |
| CA | same rational family with continuation on the exponent |
| STO/DA/SOF/TY | standard porous penalty / not present |

**Recommendation:** keep our Borrvall–Petersson rational form `α_min + (α_max−α_min)·q·ρ/(q+1−ρ)`
(equivalent formulation) — it is the field standard. Tune `α_max` (≈10⁴–10⁶) and penalty `q` using TF's
defaults as a starting point; apply continuation on `q` for sharper designs.

### 3. Density / design-variable representation

All density-based codes (TF, STO, CA, TY) use **one design variable per cell** mapped through filter →
projection → material interpolation. TY additionally shows multi-material extensions.

**Recommendation:** keep our **cell-centered ρ ∈ [0,1] over the design mask** (tray/plenum regions only).
This matches every reference and keeps the gradient assembly simple.

### 4. Forward PDE solve

| Repo | Approach |
|---|---|
| TF | Newton on residual + analytic Jacobian (`RES.m`/`JAC.m`) |
| CA | direct/iterative linear solves per FEniCS assemble |
| STO | GPU linear algebra |
| TY | none (no PDE) |

**Recommendation:** keep our **Newton + line search with sparse direct solves**
(scipy.sparse / SuperLU-style). TF validates this choice for exactly this class of problem. Only revisit
if 3D/GPU becomes necessary (STO's route).

### 5. Adjoint solve

| Repo | Approach |
|---|---|
| TF | **Discrete adjoint**: solve `Jᵀλ = ∂J/∂u` (transposed linearized operator) |
| DA | Automated tape → transposed block solve (same math, generated) |
| SOF | **Continuous adjoint**: solve the adjoint PDE (`qEqn.H`) with adjoint BCs |
| CA | Adjoint/topological derivative per problem |
| STO | GPU discrete adjoint gradient |

**Recommendation:** keep our **discrete adjoint** (transposed Jacobian) — it is exact, matches TF, and is
the right choice for a self-contained solver. Validate it against DA's pattern and our own finite
differences (we already have an FD test). Use SOF's continuous-adjoint treatment of **inlet/outlet
adjoint BCs** only as intuition for the pressure-cap chain rule.

### 6. Sensitivity / gradient calculation

| Repo | Approach |
|---|---|
| TF | `∂J/∂ρ = λ·∂R/∂ρ + ∂J/∂ρ` via `dRESdg`, `dPHIdg`, `dPHIds` |
| TY | same chain rule for structural objectives |
| DA/CA | automatic / adjoint-assembled |

**Recommendation:** keep our chain rule through the **adjoint state and the filter** (already
implemented). TF's split into "flow residual derivative" and "filter derivative" is the structure to
maintain for testability. For our fan/plate objectives: differentiate **through the fan flow `Q`** (the
signed `dQ/du` chain rule we already fixed) — that is our project-specific sensitivity that none of the
references cover directly, because they model fixed inlet/outlet flows, not internal fans.

### 7. Density filtering

| Repo | Approach |
|---|---|
| TF | convolution-style neighborhood filter in `PHI.m` |
| CA | Helmholtz-PDE filter (implicit, FEniCS-native) |
| TY | standard radius filter |
| STO/SOF | — |

**Recommendation:** keep our **explicit convolution/radius filter** (filters.py) — simplest on a
structured grid, matches TF/TY. Ensure the gradient is passed through the same filter (transposed
operation) — already the case. Radius ≈ 2–4 cells balances length-scale control vs. feature resolution.

### 8. Projection / thresholding

| Repo | Approach |
|---|---|
| CA | smooth **tanh projection** with continuation (best-documented) |
| TY | discrete thresholding during updates |
| TF | mild/intermediate design snapshots |

**Recommendation:** adopt CA's **smooth tanh projection with continuation** (projection strength increases
over iterations) to drive near-0/1 designs while keeping gradients smooth. Threshold at ρ=0.5 only at the
very end (manufacturability step, not optimization step).

### 9. Volume constraints

| Repo | Approach |
|---|---|
| TY/TF | **OC with Lagrange multiplier** satisfying the volume fraction exactly |
| CA | constraints inside an SQP/MMA-style loop |
| POPOVAC (paper) | explicit volume-preserving update |

**Recommendation:** our penalty-on-solid-fraction is the pragmatic v1 and works. The reference-backed
upgrade is **OC with a volume-fraction constraint** (TF/TY pattern): single-constraint case solved by
bisection on the Lagrange multiplier. Upgrade when the penalty causes target drift.

### 10. Pressure-drop / objective constraints

| Repo | Approach |
|---|---|
| CA | NS **pipe-bend min pressure-drop** objective (closest analog to our plenum pressure) |
| TF | inlet–outlet pressure-drop objectives (diffuser cases) |
| SOF | objective functions with adjoint BCs |

**Recommendation:** our combined **uniformity objective + pressure-cap penalty** is project-specific;
keep it, but borrow CA's approach for how the pressure term enters the objective and how its gradient is
consistently differentiated (we already fixed the `dJ_p/dQ · dQ/du` sign chain). Treat the pressure cap as
a soft penalty now; promote it to a proper constraint (augmented Lagrangian) only if it fights the
uniformity objective in runs.

### 11. Optimization / update algorithm

| Repo | Approach |
|---|---|
| TF | OC + continuation (robust, monotone for single constraint) |
| TY | OC and **MMA** implementations |
| CA | L-BFGS / truncated Newton with line search |

**Recommendation:** keep our gradient-based update; standardize on **OC-style with volume constraint +
continuation** (TF/TY) for the 1-constraint regime. Move to **MMA** (re-implemented or via a small pure-
Python MMA, e.g. nlopt-adjacent) when we run pressure + volume constraints simultaneously. BFGS (CA) is
an alternative only if iteration counts balloon; our warm-started Newton makes it unnecessary.

### 12. Geometry extraction

| Repo | Approach |
|---|---|
| TY/TF | threshold → surface/STL export |
| CA | mesh manipulation on FEniCS grids |

**Recommendation (2D path):** threshold ρ=0.5 → **marching-squares contour**
(skimage.measure.find_contours or scipy) → polygon cleanup (shapely) → **DXF export** → extrude to 3D.
This is the structured-grid analogue of TF/TY's STL export and plugs directly into our CAD pipeline.

### 13. Visualization

| Repo | Approach |
|---|---|
| TF | MATLAB field plots (velocity/pressure/density) |
| TY | Mayavi 3D |
| CA | VTK/Paraview |
| SOF | ParaView |

**Recommendation:** keep our **matplotlib 2D field plots** (plots.py) — velocity, pressure, density +
overlay of tray rows. Add a **binary-density overlay** (from the projection step) for the manufacturable
design view. Matches TF's 2D post-processing style.

### 14. Performance optimization

| Repo | Approach |
|---|---|
| STO | **CUDA GPU** linear algebra + GPU gradient |
| TF | analytic elements + sparse assembly |
| CA/DA | PETSc/MPI |

**Recommendation:** stay with **sparse direct solves + vectorized numpy assembly** for 2D (fast enough —
our 48×112 solves run in ~2–3 min). Reuse Jacobian factorization across adjoint solve (one LU, two
triangular solves) — a cheap, high-value optimization. STO's GPU route is the reference if we ever scale
to large 3D.

---

## Recommended stack summary

| Component | Recommendation | Source of the idea |
|---|---|---|
| Flow solver | MAC-grid FV Stokes/Brinkman (keep) | ours + TF as benchmark |
| Brinkman | Borrvall–Petersson rational + continuation on q | TF, CA, papers |
| Design variables | cell-centered ρ over design mask | all |
| Forward solve | Newton + line search, sparse direct | ours = TF pattern |
| Adjoint | discrete (transposed Jacobian) + FD test | TF, DA (validation) |
| Sensitivity | chain rule incl. filter and fan-flow Q terms | TF structure + our fix |
| Filtering | explicit radius/convolution filter | TF, TY |
| Projection | tanh projection with continuation | CA |
| Volume constraint | OC with multiplier (v2 upgrade) | TF, TY |
| Pressure cap | soft penalty now; augmented-Lagrangian later | CA pipe-bend |
| Update | OC/MMA | TF, TY |
| Geometry | ρ=0.5 contour → DXF → extrude | TY/TF export philosophy |
| Visualization | matplotlib 2D fields + binary overlay | TF postproc |
| Performance | sparse reuse, vectorized numpy; GPU = future | STO |

**Bottom line.** The reference survey confirms our existing architecture (structured FV + discrete adjoint +
filtering + penalty volume control) is the right family. The concrete upgrades it motivates are:
(1) **OC-style volume constraint** to replace the soft solid-fraction penalty, (2) **tanh projection with
continuation** for manufacturable 0/1 designs, and (3) **Reynolds continuation** as we approach NS. None
of these require depending on the reference codebases.
