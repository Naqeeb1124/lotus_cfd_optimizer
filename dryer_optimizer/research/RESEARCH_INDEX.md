# Research Index — Reference Repositories for the Dryer Topology Optimizer

This index explains the role of each reference repository in the `dryer_optimizer` project and maps each
one to the 14 technical components of our 2D fluid topology optimization pipeline.

> These repositories are **research/reference only** (see `reference_code/REFERENCE_MANIFEST.md`).
> The production solver must not depend on them. They exist to answer "how do the proven implementations
> do this?" while we tune the math, physics, and engineering of our own solver.

---

## Roles at a glance

| Repository | Role in our project |
|---|---|
| **topflow** (MATLAB, BSD-3) | **Primary blueprint.** End-to-end density-based flow topology optimization: Stokes/NS, Brinkman penalization, discrete adjoint, OC update, continuation. The workflow our Python solver mirrors; the reference we check our gradients against. |
| **Stokes_TO** (C++/CUDA, unlicensed) | **Performance reference.** Shows how to scale the flow solve + gradient to big grids via GPU linear algebra. Only relevant later, if we outgrow 2D structured grids. |
| **cashocs** (Python/FEniCS, GPL-3.0) | **Objective-engineering reference.** The pipe-bend demo is Navier–Stokes topology optimization for **minimum pressure drop** — the closest published analog to our fan/plenum pressure handling. Also the best reference for filter + tanh-projection + continuation recipes. |
| **dolfin-adjoint** (Python/FEniCS, LGPL-3.0) | **Adjoint-correctness reference.** Demonstrates the discrete-adjoint pattern (adjoint = transpose of the linearized forward operator) in its purest automated form. We use the *pattern*, not the code, to validate our hand-derived adjoint. |
| **topy** (Python, GPL-3.0) | **Update/geometry reference.** Pure-Python OC/MMA updates, density filtering, and STL export — a dependency-free template for the update and manufacturable-geometry stages of our Python pipeline. (Structural only — no flow physics.) |
| **shapeOptimizationFoam** (C++/OpenFOAM, GPL-3.0) | **Continuous-adjoint & 3D-validation reference.** The other school of adjoint (continuous PDE adjoint vs. our discrete adjoint), and the closest relative to the OpenFOAM/Fluent 3D validation we plan after the 2D pipeline. |

---

## Component mapping (14 components)

Legend: ✅ implements/demonstrates directly · ◐ partial / related · ✗ absent

| # | Component | topflow | Stokes_TO | cashocs | dolfin-adjoint | topy | shapeOptimizationFoam |
|---|-----------|:-------:|:---------:|:-------:|:--------------:|:----:|:---------------------:|
| 1 | Stokes flow | ✅ NS + Stokes continuation | ✅ GPU Stokes | ✅ NS (pipe bend) | ◐ any FEniCS PDE | ✗ (structural) | ✅ RANS + adjoint |
| 2 | Brinkman penalization | ✅ `PHI.m` (α_min→α_max, rational) | ✅ `FluidTopoOpt` | ✅ topology machinery | ◐ canonical examples | ✗ | ✗ (shape, not topology) |
| 3 | Density/design-variable representation | ✅ cell densities | ✅ density field | ✅ cell densities | ◐ | ✅ SIMP densities | ✗ |
| 4 | Forward PDE solve | ✅ Newton (`RES.m`/`JAC.m`) | ✅ GPU solver | ✅ FEniCS solves | ✅ taped solves | ✗ | ✅ OpenFOAM FV |
| 5 | Adjoint solve | ✅ discrete adjoint (transposed Jacobian) | ✅ `FluidEnergyGrad.cu` | ✅ adjoint + topological derivative | ✅✅ automated tape adjoint | ◐ adjoint routines | ✅✅ continuous adjoint PDE |
| 6 | Sensitivity/gradient | ✅ `dRESdg`, `dPHIdg`, `dPHIds` | ✅ GPU gradient | ✅ via adjoint | ✅ automatic | ✅ sensitivity assembly | ✅ shape gradient |
| 7 | Density filtering | ✅ `PHI.m` filter | ◐ | ✅ Helmholtz-style filter | ◐ | ✅ filter radius | ✗ |
| 8 | Projection/thresholding | ◐ | ◐ | ✅ tanh projection + continuation | ◐ | ✅ thresholding in updates | ✗ |
| 9 | Volume constraints | ✅ OC + volume constraint | ◐ | ✅ constraint handling | ◐ | ✅ OC volume constraint | ✗ |
| 10 | Pressure-drop / objective constraints | ✅ diffuser pressure-drop objectives | ✅ energy objective | ✅✅ pipe-bend pressure drop | ◐ | ✗ | ✅ objective functions |
| 11 | Optimization/update algorithm | ✅ OC + continuation | ✅ steepest/gradient descent | ✅ L-BFGS, truncated Newton | ◐ | ✅ OC / MMA | ✅ steepest descent |
| 12 | Geometry extraction | ✅ `export.m` (STL) | ◐ | ✅ mesh output | ◐ | ✅ STL export | ✅ mesh/CAD workflow |
| 13 | Visualization | ✅ `postproc.m` | ✅ GPU viewer | ✅ VTK/Paraview | ◐ | ✅ Mayavi | ✅ ParaView |
| 14 | Performance optimization | ✅ analytic elements, sparse | ✅✅ CUDA | ✅ PETSc/MPI | ✅ PETSc | ◐ | ✅ parallel OpenFOAM |

---

## How each component informs our solver

1. **Stokes flow** — topflow (NS with continuation) and cashocs (pipe-bend NS) confirm our Stokes/Brinkman
   forward solver and give benchmark cases; Stokes_TO shows the high-performance route.
2. **Brinkman penalization** — our `alpha_min + (alpha_max−alpha_min)·q·ρ/(q+1−ρ)` matches Borrvall–Petersson
   as implemented in topflow's `PHI.m`; keep ours, validate the constants against topflow.
3. **Density representation** — all density-based codes use per-cell design variables; ours is consistent.
4. **Forward PDE solve** — our Newton + line search mirrors topflow's `RES.m`/`JAC.m` pattern; when we add
   continuation we copy topflow's schedule (low Reynolds → target).
5. **Adjoint solve** — our discrete adjoint (transposed Jacobian) is the topflow/dolfin-adjoint pattern.
   dolfin-adjoint is our correctness oracle for small validation cases.
6. **Sensitivity** — our chain rule `dJ/dρ = dJ/du · du/dρ` matches topflow's `dRESdg`/`dPHIdg` split.
7. **Density filtering** — we already filter (filters.py); cashocs and topflow justify radius choice and
   gradient-smoothing. (Our density gradient must be passed through the same filter — already done.)
8. **Projection/thresholding** — cashocs' tanh projection with continuation is the recommended route to
   near-0/1 designs for manufacturability; topy does plain thresholding.
9. **Volume constraints** — topy's OC volume constraint and topflow's OC both handle the single-constraint
   case; our penalty-based solid-fraction control is the pragmatic v1 (see research papers on Popovac for
   the exact-constraint upgrade).
10. **Pressure-drop / objective constraints** — cashocs' pipe-bend NS objective is the reference for our
    pressure-cap gradient (the `dJ_p/dQ` chain rule we fixed); shapeOptimizationFoam shows the continuous
    adjoint treatment of objective BCs.
11. **Update algorithm** — OC (topflow/topy) is the simple robust choice for one constraint; MMA (topy)
    is the upgrade path when we add pressure + volume constraints simultaneously.
12. **Geometry extraction** — topy/topflow export STL; for us: contour ρ=0.5 on the structured grid →
    polygon → DXF → extrusion to 3D (matches topy's post-processing philosophy).
13. **Visualization** — we use matplotlib (plots.py); topflow's postproc.m is the 2D-field precedent.
14. **Performance** — Stokes_TO (CUDA) is the far-future path; for now, sparse direct solves on the
    structured grid (as in topflow's analytic-element + sparse pattern) are sufficient for 2D.

---

## Decision rule

- **Adopt the approach** of topflow (discrete adjoint + OC + continuation) as the backbone — it is the same
  family as our existing solver, permissively licensed, and easiest to validate.
- **Borrow recipes** (not code) from cashocs for filtering/projection and pressure-drop objectives.
- **Ignore for production** the FEniCS-based tools (cashocs, dolfin-adjoint) and GPU/C++ tools
  (Stokes_TO, shapeOptimizationFoam, OpenFOAM) — they serve only as conceptual/validation/performance
  references. Final production solver stays a standalone Python structured-grid code.
