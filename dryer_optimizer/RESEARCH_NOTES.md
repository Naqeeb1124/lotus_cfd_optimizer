# Research notes for V1

The six priority papers are stored locally in `research_papers/`. The current
implementation uses their shared mathematical direction, while deliberately
keeping the first model small enough for verification.

| Priority paper | V1 decision informed by it |
|---|---|
| Othmer (2008), *A Continuous Adjoint Formulation for the Computation of Topological and Surface Sensitivities of Ducted Flows* | Treat the design sensitivity as a derivative of the flow residual and keep the adjoint boundary/system formulation consistent with the forward discretization. |
| Othmer, de Villiers & Weller (2007), *Implementation of a Continuous Adjoint for Topology Optimization of Ducted Flows* | Use a Brinkman resistance field to represent implicit fluid/solid topology rather than remeshing every candidate baffle. |
| Popovac (2022), *Continuous Adjoint Topology Optimization of Duct Flow Configurations with Explicit Volume Constraint for Design Variable Update* | Include an explicit design-volume penalty/limit in addition to the tray-uniformity objective, preventing a purely flow-based but impractical solution. |
| Vrionis, Samouchos & Giannakoglou (2021), *Topology Optimization in Fluid Mechanics Using Continuous Adjoint and the Cut-Cell Method* | Preserve immutable walls, tray neighborhoods, inlet, and outlet separately from the design field; future cut-cell work can replace the current structured-mask treatment. |
| Pimanov et al. (2025), *Sparse Narrow-Band Topology Optimization for Large-Scale Thermal-Fluid Applications* | Keep the design mask and sparse operator interfaces explicit so a later narrow-band/scalable implementation can avoid assembling inactive regions. Thermal coupling is intentionally not in V1. |
| Hirotani et al. (2026), *Topology Optimization of Cooling Channels Using Dual-Type Moving Morphable Components* | Treat manufacturing representation as a downstream concern: V1 exports a clean 2D topology, while a component/feature-based representation can be added for CAD-ready baffle families later. |

## What is implemented now

- linear steady incompressible Brinkman/Stokes flow on a staggered MAC grid;
- density-to-resistance penalization;
- tray-average normal velocity and coefficient-of-variation objective;
- pressure-drop and explicit baffle-material penalties;
- exact discrete transpose adjoint;
- finite-difference gradient verification;
- sparse density filtering, smooth projection, move limits, and SVG/NPZ output;
- CAD-derived 20-tray, 1.630 m dimensional constraints.

## What remains deliberately outside V1

The current model is not a replacement for the cited full formulations or for
3D CFD validation. It does not yet include nonlinear convection, cut-cell
boundary geometry, turbulence, temperature, humidity, species transport,
transient effects, narrow-band acceleration, moving morphable components,
SolidWorks automation, or Fluent automation. Those additions should follow a
successful 2D gradient and baseline-vs-optimized validation study.
