# betaflow

A solver-agnostic validation harness for haemodynamic CFD. Each case pairs an
analytic oracle with a solver run and asserts an explicit error norm against an
explicit tolerance, logging a provenance-stamped record. It is a regression
suite, not a one-off check: rerun it after any solver upgrade, scheme change,
or new boundary-condition library, and diff the committed results.

## What "validated" means here

A case is validated when, for a stated Reynolds number and mesh level,

    metric(numerical profile, analytic oracle) < tol

with the metric, tolerance, and non-dimensionalisation all declared in the
case YAML — never inside a runner. The committed `results/*.json` records the
error, mesh level, cell count, solver version, git SHA, and timestamp, so any
future regression is attributable to a specific change. A failing case is
diagnosed (oracle vs mesh vs boundary conditions), never "fixed" by loosening
the tolerance.

Two kinds of verification, deliberately distinguished (reviewers notice):

- **Code verification** — the case has an exact solution. The harness measures
  the *true* discretisation error and runs an **order-of-accuracy test**:
  observed p = log2(e_coarse/e_fine) must match the formal order of the
  scheme. This is stronger than any error-estimation procedure. All analytic
  rungs (Poiseuille, later Womersley) get this treatment;
  see `report/order_of_accuracy.md`.
- **Solution verification** — no exact solution (Carreau rheology, patient
  geometry). There the discretisation error can only be *estimated*, and GCI
  enters. GCI is never used where an oracle exists.

Non-dimensionalisation is where validation harnesses silently rot. The
Reynolds-number definition (bulk velocity, full channel height:
`Re = u_mean * 2h / nu`) is stated once in the oracle docstring
(`betaflow/analytic/poiseuille.py`), echoed in the case YAML, and
cross-checked by the test against the viscosity the runner actually used.

## Layout

    betaflow/
      analytic/   oracles — pure functions, no solver knowledge
      cases/      YAML case definitions (geometry, Re, oracle path, metric+tol)
      runners/    solver adapters — the ONLY layer that knows a solver exists
      metrics/    error norms over plain arrays
    tests/        pytest, one test per case
    results/      logged JSON records, committed

The design constraint: nothing above `runners/` may know OpenFOAM exists.

    run_case(case, runner="openfoam", n_cells=80)
      -> {"y": ndarray, "u": ndarray, "u_ref": float, "meta": {...}}

Metrics and tests consume only that dict. `u_ref` is the velocity the oracle
normalises by (for Poiseuille, the analytic peak velocity); `meta` carries
provenance (solver version, cell counts, viscosity) and is never used for
physics.

## Running

    python3 -m pytest tests/ -v

The runner sources `/opt/openfoam14/etc/bashrc` itself; set
`BETAFLOW_OPENFOAM_BASHRC` to point elsewhere (the templates use OpenFOAM 14
Foundation syntax, so older versions will fail at dictionary parse).

Generated OpenFOAM cases land in `_runs/` (gitignored) for inspection.

## Current cases

| case | metric | oracle | mesh level | error | tol |
|---|---|---|---|---|---|
| poiseuille_steady | L2_velocity | plane Poiseuille parabola | 80 cells across 2h | 3.25e-4 | 1e-3 |
| poiseuille_steady | wss_relative | tau_w = G h (kinematic) | 80 cells across 2h | 3.12e-4 | 2e-2 |

Order-of-accuracy test (mesh levels 40/80/160, refinement ratio 2, logged in
`results/poiseuille_steady_refinement.json`, tabulated in
`report/order_of_accuracy.md`):

| sampling | errors (40/80/160) | observed p |
|---|---|---|
| lineUniform + cellPoint interpolation | 1.30e-3 / 3.25e-4 / 8.13e-5 | 1.999, 2.000 |
| lineCell raw cell values | 4.18e-4 / 1.05e-4 / 2.65e-5 | 1.986, 1.991 |

Both sampling modes converge at the formal order (accepted band 1.8–2.2,
asserted by `tests/test_refinement.py`). Profile interpolation contributes
about 3× the raw-cell error but converges at the same rate — no plateau. The
raw-cell error itself is the discrete bulk-flow normalisation effect: the
`meanVelocityForce` constraint drives the *cell-centre mean* to Ubar, giving
the exact parabola plus a uniform dy²/4 offset with a correspondingly adjusted
pressure gradient (predicted RMS 4.3e-4 at N=40; measured 4.18e-4).

Wall shear stress also converges at p = 2.0 (1.25e-3 / 3.12e-4 / 7.81e-5) —
but NOT because the wall gradient is high-order. The functionObject uses the
plain half-cell one-sided snGrad, formally O(dy). At steady state the discrete
global momentum balance forces the wall flux to equal the imposed mean
pressure-gradient source exactly (tau_num = G_disc * h to all digits of the
solver log), and G_disc itself converges at O(dy²) — the O(dy) truncation of
the one-sided difference cancels against the dy²/4 offset the discrete
solution carries. This identity is a property of force-driven periodic flow;
it will NOT hold for transient (Womersley) or non-periodic cases, where the
wall-gradient order gets tested for real.

A hand-picked tolerance is arbitrary; the observed order is what makes an
error number meaningful. The N=40 cellPoint error (1.30e-3) *fails* the 1e-3
tolerance — expected, and the reason the per-case tolerance is tied to a
stated mesh level.

Cross-version check: the L2 error is identical to all logged digits under
OpenFOAM 12 and OpenFOAM 14 (see the results history in git) — the upgrade
changed dictionary syntax, not the discrete solution.

## Adding a case

1. Write the oracle in `betaflow/analytic/` — a pure function returning
   non-dimensional profile values, with the Reynolds-number definition (length
   AND velocity scale) stated in the docstring.
2. Add `betaflow/cases/<name>.yaml`: geometry, `nondim`, dotted `oracle` path,
   `normalisation`, and `metrics: [{name, tol}]`.
3. Extend the runner(s) to set up that geometry (add a template set if needed).
4. Add `tests/test_<name>.py`: load YAML, `run_case`, evaluate metric, assert,
   write `results/<name>.json`.

## Adding a runner

Add one module `betaflow/runners/<solver>.py` exposing
`run(case, n_cells=..., **params)` that returns the standard dict. Nothing
else changes — `run_case(case, runner="<solver>")` dispatches by module name.
Keep every solver-specific file (dictionaries, templates, log parsing) inside
that module and its template directory; if a metric or test needs to import
it, the layering is broken.

## OpenFOAM runner notes

OpenFOAM 14 (Foundation), `foamRun` with the `incompressibleFluid` module.
Channel cases are one cell thick in z (`empty` front/back), cyclic streamwise
patches, driven by a `meanVelocityForce` fvConstraint so there are no
entrance-length effects and the domain stays short. Profiles are sampled with
the `sets` functionObject via `foamPostProcess`, in one of two modes:
`sampling="cellPoint"` (default; fixed lineUniform stations, cellPoint
interpolation — what a user extracting a profile gets) or `sampling="cell"`
(lineCell raw cell values — the discrete solution with no interpolation
error). Wall shear stress is computed by the `wallShearStress` functionObject
during the solve (`system/functions` is auto-read, so the momentum-transport
model it needs exists) and parsed from the written field; incompressible
OpenFOAM reports it in kinematic units [m²/s²]. All dictionaries are rendered
from `string.Template` files in `runners/openfoam_templates/` — mesh level,
viscosity, and forcing are parameters, never hand edits.

Steady convergence is gated on the parsed Ux initial residual (< 1e-9), not on
`residualControl`: the wall-normal velocity and pressure are identically zero
in this flow, so their normalised residuals stagnate at O(0.1) no matter how
converged the run is.
