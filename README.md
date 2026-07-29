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

| case | oracle | mesh level | L2 error | tol |
|---|---|---|---|---|
| poiseuille_steady | plane Poiseuille parabola | 80 cells across 2h | 3.25e-4 | 1e-3 |

The error is pure second-order discretisation + profile-interpolation error:
5.18e-3 / 1.30e-3 / 3.25e-4 at 20/40/80 cells across the channel — ratio 4.0
per doubling. Mesh level is a first-class parameter of every runner, so the
three-level GCI study needs no refactor, just three calls.

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
the `sets` functionObject via `foamPostProcess`. All dictionaries are rendered
from `string.Template` files in `runners/openfoam_templates/` — mesh level,
viscosity, and forcing are parameters, never hand edits.

Steady convergence is gated on the parsed Ux initial residual (< 1e-9), not on
`residualControl`: the wall-normal velocity and pressure are identically zero
in this flow, so their normalised residuals stagnate at O(0.1) no matter how
converged the run is.
