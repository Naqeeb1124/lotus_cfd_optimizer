# Reference Code Manifest

Local **read-only** copies of public reference repositories, acquired to support the `dryer_optimizer`
project (2D fluid topology optimization for multi-tray industrial dryer airflow uniformity).

These copies exist strictly for **research / reference purposes**:

- They are **not modified**.
- They are **not integrated** into the production solver.
- Each repository is isolated in its own subdirectory, with LICENSE, README, and attribution preserved.
- The cloned subdirectories are **git-ignored** (see root `.gitignore`); only this manifest is tracked.
- If a repository could not be obtained it is reported below (none failed).

Acquired: **2026-08-13** — shallow clones (`git clone --depth 1`).

| # | Repository | URL | Commit (date) | License | Language |
|---|------------|-----|---------------|---------|----------|
| 1 | topflow | https://github.com/sdu-multiphysics/topflow | `a3a3f7b` (2023-01-26) | BSD-3-Clause | MATLAB |
| 2 | Stokes_TO | https://github.com/jyl-pages/Stokes_TO | `308a79c` (2022-02-26) | **None** (see §2) | C++ / CUDA |
| 3 | cashocs | https://github.com/sblauth/cashocs | `8bdd5d7` (2026-08-11) | GPL-3.0 | Python (FEniCS) |
| 4 | dolfin-adjoint | https://github.com/dolfin-adjoint/dolfin-adjoint | `6c5bc13` (2025-12-16) | LGPL-3.0 | Python (FEniCS) |
| 5 | topy | https://github.com/williamhunter/topy | `3c461c4` (2022-08-31) | GPL-3.0 | Python |
| 6 | shapeOptimizationFoam | https://github.com/joslorgom/shapeOptimizationFoam | `474daf5` (2018-06-29) | GPL-3.0 (OpenFOAM) | C++ (OpenFOAM) |

---

## 1. topflow

- **URL:** https://github.com/sdu-multiphysics/topflow
- **Name:** topflow — Topology Optimization of Fluid Flow
- **Commit / version:** `a3a3f7bdc08d03b208710659c2831a285f898bee` (2023-01-26), shallow clone of default branch
- **License:** BSD-3-Clause (`LICENSE.md`) — permissive, safe to study and adapt
- **Original purpose:** Compact MATLAB implementation of topology optimization for **Navier–Stokes flow**
  (Alexandersen-style, 2023 companion code), covering Stokes→NS continuation, discrete adjoint, and
  optimality-criteria (OC) updates.
- **Language:** MATLAB (15 source files, no external toolbox dependency)
- **Relevant files:**
  - `topFlow.m` — main optimization driver: continuation loop, OC update, convergence
  - `problems.m` — problem definitions (design domain, BCs, objectives; incl. classic diffuser cases)
  - `RES.m` / `JAC.m` — nonlinear residual and Newton Jacobian of the flow system
  - `analyticalElement.m` — analytic element matrices (mass/stiffness, Brinkman term)
  - `PHI.m` — density→permeability interpolation and **density filter**
  - `dRESdg.m`, `dPHIdg.m`, `dPHIds.m` — discrete sensitivity chain-rule derivatives
  - `postproc.m` — field visualization
  - `export.m` — geometry export (STL-style triangulation)
- **Potentially useful to our project:** the *exact* discrete-adjoint + OC workflow our Python solver mirrors;
  the Brinkman interpolation form; analytic element assembly pattern; continuation schedule. The closest
  end-to-end reference for our 2D structured-grid solver.

---

## 2. Stokes_TO

- **URL:** https://github.com/jyl-pages/Stokes_TO
- **Name:** Stokes_TO — GPU Stokes Topology Optimization
- **Commit / version:** `308a79cf5806a23d99f9f69ed43bf11b9097b05e` (2022-02-26), shallow clone
- **License:** ⚠️ **None found for the project's own code.** The vendored third-party libraries
  (Eigen, CUSP, freeglut, etc. under `complex/`) carry their own licenses. Treat the project code as
  **all-rights-reserved / research-only** — do not copy its code into production without upstream
  clarification.
- **Original purpose:** CUDA-accelerated Stokes topology optimization ("PainlessSolver") targeting large
  3D-scale problems via GPU linear algebra.
- **Language:** C++ / CUDA (with vendored libs; 92 MB, 3592 files — mostly third-party)
- **Relevant files:**
  - `complex/proj/PainlessSolver/proj/fluid_topo/FluidTopoOpt.h/.cpp` — main fluid topology-optimization driver
  - `complex/proj/PainlessSolver/proj/fluid_topo/FluidEnergy.h/.cpp` — energy objective (flow power) assembly
  - `complex/proj/PainlessSolver/proj/fluid_topo/FluidEnergyGrad.cu` — **GPU gradient evaluation**
- **Potentially useful to our project:** GPU-parallelization strategy for the flow solve and gradient;
  design of the energy/pressure-drop objective. Architecture reference only — not for direct code reuse.

---

## 3. cashocs

- **URL:** https://github.com/sblauth/cashocs
- **Name:** cashocs — Computational Adjoint-Based Shape Optimization and Optimal Control Software
- **Commit / version:** `8bdd5d74d7427bf80cec69b76e8245294187ab02` (2026-08-11), shallow clone
- **License:** GPL-3.0 (`COPYING`) — copyleft; study/compare OK, cannot link into non-GPL production code
- **Original purpose:** FEniCS-based framework for PDE-constrained optimization (shape optimization,
  optimal control, and **topology optimization**) using adjoint methods with density filtering/projection.
- **Language:** Python (FEniCS), 540 files; demos + `cashocs/` package
- **Relevant files:**
  - `cashocs/_pde_problems/` — topology-optimization PDE machinery (filtering, projection, topological derivative)
  - `demos/documented/topology_optimization/pipe_bend/demo_pipe_bend.py` — Navier–Stokes topology optimization
    for minimal pressure drop (our closest analog: pressure-drop objective)
  - `demos/documented/topology_optimization/pipe_bend/config.ini` — optimizer config (BFGS line search, linear system flag)
- **Potentially useful to our project:** the **pipe-bend NS pressure-drop objective** formulation; the
  Helmholtz-style density filter and tanh projection with continuation; BFGS/truncated-Newton update options.

---

## 4. dolfin-adjoint

- **URL:** https://github.com/dolfin-adjoint/dolfin-adjoint
- **Name:** dolfin-adjoint (package: `fenics_adjoint`)
- **Commit / version:** `6c5bc137051058b7397acf22f4300c87f36c317c` (2025-12-16), shallow clone
- **License:** LGPL-3.0 (`LICENSE`) — permissive for dynamic linking, but FEniCS-bound
- **Original purpose:** Automatic (tape-based) differentiation of FEniCS PDE solves: records every solve on
  a tape and assembles the adjoint automatically, so any PDE-constrained objective yields its gradient
  without hand-derived adjoint equations.
- **Language:** Python (FEniCS)
- **Relevant files:**
  - `src/fenics_adjoint/` — tape recording, annotation, adjoint/forward replay
  - `src/fenics_adjoint/blocks/` — per-operation blocks (solve blocks, function blocks) driving the tape
- **Potentially useful to our project:** conceptual reference for the *discrete adjoint* correctness pattern
  (adjoint = transposed linearized operator) and for validation of our hand-derived adjoint on small cases.
  Not usable in production: our solver is a custom structured-grid FV code, not FEniCS.

---

## 5. topy

- **URL:** https://github.com/williamhunter/topy
- **Name:** topy — Topology Optimization with Python
- **Commit / version:** `3c461c4b65c0f5f0c5476a4711a26a7e1ff64e58` (2022-08-31), shallow clone
- **License:** GPL-3.0 (`LICENSE.md`) — copyleft; reference only
- **Original purpose:** Pure-Python structural topology optimization (SIMP with multiple material models)
  and objective classes; features filtering, OC/MMA-style updates, and STL export.
- **Language:** Python (no structural flow solver — **structural only**)
- **Relevant files:**
  - `topy/topy/topology.py` — `Topology` class: design variables, filtering, sensitivity assembly
  - `topy/topy/optimisation.py` — OC/MMA update schemes
  - `topy/topy/elements.py` — element/material model library
  - `topy/examples/` — runnable 2D/3D demos
- **Potentially useful to our project:** a pure-Python **reference for density filtering, volume-constraint
  handling, and OC/MMA update code** that we can compare against our update.py without FEniCS/CUDA
  dependencies. Its fluid capability is nil — do not look to it for the flow physics.

---

## 6. shapeOptimizationFoam

- **URL:** https://github.com/joslorgom/shapeOptimizationFoam
- **Name:** shapeOptimizationFoam — OpenFOAM continuous adjoint shape optimization solver
- **Commit / version:** `474daf54a317e7b51d91a3568957ecef596ddda4` (2018-06-29), shallow clone
- **License:** GPL-3.0 (no standalone LICENSE file; OpenFOAM-standard GPL-3.0 header on every source file)
- **Original purpose:** A standalone OpenFOAM solver demonstrating **continuous adjoint shape optimization**
  for (turbulent) flow — adjoint momentum/continuity equations solved as PDEs with adjoint BCs.
- **Language:** C++ (OpenFOAM)
- **Relevant files:**
  - `shapeOptimizationFoam.C` — main solver loop (primal + adjoint)
  - `qEqn.H` — **adjoint momentum equation** (continuous adjoint PDE)
  - `adjointBC.H` — adjoint boundary conditions
  - `createPsi.H` — adjoint variable creation
  - `Make/` — wmake build files
- **Potentially useful to our project:** the *continuous-adjoint* counterpart to topflow's discrete adjoint
  (validates our choice of discrete adjoint for a structured grid); and the adjoint-BC treatment of
  inlets/outlets — relevant to the pressure-cap gradient we fixed (chain rule through the pressure penalty).
  Also the eventual bridge to 3D Fluent validation (OpenFOAM-adjacent workflow).

---

## Obtained / failed summary

All six repositories were cloned successfully — **no failures**. Provenance (commit + date) is pinned
above for reproducibility. License caveats: **Stokes_TO has no license on its own code** (research-only),
and **shapeOptimizationFoam relies on the GPL-3.0 header** convention; both are fine as read-only
references but cannot be absorbed into non-GPL production code without care.
