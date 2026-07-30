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

    python3 -m pytest tests/ -v          # ~15 min; casson_steady is ~9 of it
    python3 -m pytest tests/ -v --ignore=tests/test_casson.py   # ~5 min

The runner sources `/opt/openfoam14/etc/bashrc` itself; set
`BETAFLOW_OPENFOAM_BASHRC` to point elsewhere (the templates use OpenFOAM 14
Foundation syntax, so older versions will fail at dictionary parse).

Generated OpenFOAM cases land in `_runs/` (gitignored) for inspection.

## Current cases

| case | metric | oracle | mesh level | error | tol |
|---|---|---|---|---|---|
| poiseuille_steady | L2_velocity | plane Poiseuille parabola | 80 cells across 2h | 3.25e-4 | 1e-3 |
| poiseuille_steady | wss_relative | tau_w = G h (kinematic) | 80 cells across 2h | 3.12e-4 | 2e-2 |
| couette_steady | L2_velocity | linear Couette profile | 40/80/160 (null test) | 0 / 0 / 1.6e-14 | 1e-8 |
| couette_steady | wss_relative | tau_w = nu u_wall / H | 40/80/160 (null test) | 0 / 0 / 5.0e-13 | 1e-8 |
| womersley_pulsatile | L2_amplitude | plane-channel Womersley (complex cosh) | alpha=20, 8 cells/delta | 8.9e-4 | 1e-2 |
| womersley_pulsatile | L2_phase [rad] | same, amplitude-weighted | alpha=20, 8 cells/delta | 2.9e-4 | 2e-2 |
| womersley_pulsatile | wss_amp_relative | tau_hat = G h tanh(K)/K | alpha=20, 8 cells/delta | 4.2e-4 | 2e-2 |
| casson_steady | L2_velocity | Casson channel profile | N=160, nuMax/nu_c=1e2 | 7.3e-4 | 1e-2 |
| casson_steady | wss_relative | tau_w = G h (rheology-independent) | N=160, nuMax/nu_c=1e2 | 3.8e-11 | 1e-6 |

womersley_pulsatile's mesh level is CELLS PER STOKES LAYER (mesh refines
with alpha): a fixed-mesh alpha-sweep cannot distinguish high-alpha error
amplification from under-resolution. Its convergence notion is
cycle-to-cycle periodicity (tol 1e-6, cap 10 cycles, logged per cycle), not
residual-to-steady; runs start from the analytic t=0 profile, without which
the O(1) startup transient decays on the (2/pi^3) alpha^2 T homogeneous-mode
timescale and no cap is meaningful. The plane-channel oracle kernel is the
complex cosh — NOT the J0 Bessel function, which is the circular-pipe
Womersley solution; see betaflow/analytic/womersley.py.

couette_steady is a NULL TEST: the exact profile is linear and every operator
in the chain (second-order interior scheme, half-cell wall gradient, linear
cellPoint interpolation) is exact for linear fields, so all errors must sit
at round-off at every level and both sampling modes — there is no
discretisation error to converge. Its tolerances are round-off budgets
(convergence gate + 12-digit ASCII I/O), and any deviation is a bug, not a
mesh effect. It earned its keep immediately: the first run exposed that the
U linear solver's absolute tolerance (1e-9) silently stalled outer
convergence three decades above the round-off floor — a floor the
force-driven Poiseuille case had partially masked. Fixing it (tolerance
1e-16, maxIter cap, iteration budget scaling with the diffusive slow mode
~N²) also drove the Poiseuille conservation-identity deviation from 5e-12 to
exactly 0.

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
plain half-cell one-sided snGrad, formally O(dy). The mechanism: the discrete
solution is the exact parabola plus a uniform offset c = G dy²/(8 nu) — the
interior equations only see differences, and c is fixed by closing the
wall-cell balance. Feeding that field into the half-cell gradient cancels the
-G dy/4 truncation term identically, so tau_num = G_disc·h to all digits of
the solver log (the conservation identity), and the entire tau error is
G_disc vs G: exactly (2/N²)/(1+2/N²), which the measurements match at every
level. The refinement test locks all three layers: p in [1.9, 2.1], the error
model to 5%, and the identity to 1e-9.

The general lesson, derived once here with numbers attached: **the formal
order of an operator is not the observed order of a conserved quantity
computed from the discrete solution.** Conservation constrains what the
discrete solution can be; in a conservative FV scheme the wall flux is pinned
by the global balance, so a formally first-order wall gradient can return a
flux with no first-order error at all.

### When error-accumulation reasoning applies — and when it does not

Two a-priori predictions were made here from operator-level reasoning, and
both were wrong the same way: **quantities linked by a discrete balance were
treated as if independently computed.** They are not. The balance is an
identity on the discrete solution, so errors in the linked quantities are
perfectly correlated by construction — they cannot accumulate independently,
and cancellation analysis does not apply.

| prediction | reasoning | outcome |
|---|---|---|
| steady WSS p ≈ 1 | one-sided wall gradient is formally O(dy) | wrong — p = 2.00, balance pins the wall flux |
| unsteady WSS error ~alpha× | tau_w is a small difference of two large terms | wrong — flat in alpha, balance pins it per timestep |

The transferable rule: **catastrophic cancellation is a real hazard for
non-conservative post-processing estimates built from separately measured
terms — not for a solver's own conserved flux.** The distinction has direct
clinical bite. WSS from 4D flow MRI *is* the hazardous case: a gradient
reconstructed from independently noisy velocity measurements, with no
balance locking it to anything, so the ~1/alpha amplification predicted above
is a genuine concern there. The same solver-side immunity should be expected
— and checked, not assumed — for every other balance-locked quantity:
branch flow splits at a bifurcation, pressure drop across a stenosis, outlet
flux distribution.

For Womersley the identity itself survives — telescoping gives
tau_w(t) = h·(G(t) − d⟨u⟩/dt) — and the a-priori prediction (p ≈ 2
preserved; WSS error ~alpha× the steady value, ~6e-3 at N=80/alpha=20,
scaling with alpha) was recorded here before the case existed. MEASURED
OUTCOME: half held. p ≈ 2 confirmed (1.93/1.97 amplitude, 2.09/2.02 WSS
under combined space-time refinement). The alpha-amplification is ABSENT:
at fixed cells-per-Stokes-layer the WSS relative error is flat in alpha
(4.10e-4 / 4.17e-4 / 4.17e-4 at alpha = 5/10/20), and the fixed-mesh
prediction point measured 1.38e-3, not 6e-3 — the increase over the
resolved value is under-resolution (2.8 cells/delta), not amplification.
The reason is the same conservation mechanism as the steady case, now
per-timestep: the discrete balance pins tau_w(t) = h(G(t) − d⟨u⟩/dt)
exactly (measured closure ≤ 4.9e-10 with the scheme-matching discrete
derivative), so tau_w is never computed as a small difference of
independently-errored large terms — catastrophic cancellation applies to
non-conservative post-processing estimates of WSS, not to the solver's own
flux. Meanwhile Euler time-stepping puts its first-order error where only a
pulsatile case can see it: velocity phase 2.4e-2 rad vs backward's 4.8e-4
at identical resolution (49×), while its WSS amplitude error crosses
fortuitously through zero and *beats* backward. Amplitude-only validation
does not merely fail to catch first-order time stepping here — it actively
prefers it. That is the concrete reason phase metrics are mandatory rather
than nice-to-have.

**Phase-metric calibration** (`phase_metric_calibration` in the results
JSON, asserted in the test). Refining time only at fixed mesh, the measured
Euler phase error matches the closed-form discrete-symbol prediction — the
scheme replaces i·omega by lambda = (1 − z)/dt, z = exp(−i·omega·dt), which
shifts both the 1/lambda prefactor and the Stokes wavenumber — to within
0.2/0.4/0.7% at n_t = 64/128/256. The leading-order lag omega·dt/2 = pi/n_t
alone accounts for 96.7% of it, constant across a 4× range; the missing 3.3%
is the profile-shape term. So the metric measures the time scheme's symbol
and nothing else. The backward arm of the same sweep does the complementary
job: its phase error plateaus at ~4.5e-4 instead of following its own
symbol prediction down, which *is* the fixed mesh's spatial phase floor —
and the reason the order study refines space and time together.

**Transient-convergence mechanism** (`decay_observed` vs `decay_predicted`
per sweep run). Once the fast Stokes-layer modes die within one cycle
(ratios drop ~50× between cycle 2 and 3), the cycle-to-cycle ratios decay
geometrically at the slowest channel mode's rate exp(−pi³/2alpha²):
predicted 0.5379 / 0.8564 / 0.9620 at alpha = 5/10/20, measured 0.5380 /
0.8564 / 0.9179 — four-digit agreement at alpha = 5 and 10, while alpha = 20
is still approaching its asymptote from below within 10 cycles (faster modes
have not fully separated). The post-fast-transient amplitude *decreases*
with alpha (8.99e-6 / 4.43e-6 / 3.08e-6), so slower decay and smaller
starting amplitude compete: `cycles_to_periodic` is non-monotonic (7 / not
met / 8), which is a measured mechanism rather than an inferred one.

A hand-picked tolerance is arbitrary; the observed order is what makes an
error number meaningful. The N=40 cellPoint error (1.30e-3) *fails* the 1e-3
tolerance — expected, and the reason the per-case tolerance is tied to a
stated mesh level.

Cross-version check: the L2 error is identical to all logged digits under
OpenFOAM 12 and OpenFOAM 14 (see the results history in git) — the upgrade
changed dictionary syntax, not the discrete solution.

## casson_steady: a non-physical parameter that changes the answer

Yield-stress viscosity is singular (nu → ∞ as shear rate → 0), so OpenFOAM
caps it: `nu = max(nuMin, min(nuMax, [sqrt(tau0/gammadot) + sqrt(m)]^2))`.
The plug is therefore never rigid — it creeps — and its computed width is set
by where nu saturates, not by the physics. That makes `nuMax` a second study
axis (`results/casson_steady.json`, 13 runs over mesh × cap ratio).

Predictions were committed before the case was built. Outcomes:

| prediction | outcome |
|---|---|
| tau_w = G_disc·h to round-off, rheology-independent | **confirmed** — exactly 0 at two grid points, ≤3.8e-11 across the converged set |
| plug-width bias = 2·sqrt(nu_c/nuMax), plug WIDER than true | **confirmed where measurable**: bias → +0.2185 at N=320 vs continuum +0.2346 |
| residual plug creep ∝ 1/nuMax | **confirmed** — factor 9.97 per decade; matches G·y_p²/(2·nuMax) to 0.3% |
| L2 error floors on nuMax, p ≈ 2 → 0 | **confirmed** — p = 1.46 then 0.04 at ratio 1e2, while ratio 1e3 still gives 1.79, 2.09 |
| measurability floor N > 1/(xi·sqrt(nu_c/nuMax)) | **confirmed, and it dominates the study** |
| practical ceiling from stiffening | **confirmed, and worse than expected** (below) |

The exact cap-active half-width is `y_p/(1 − sqrt(nu_c/nuMax))^2`, whose
leading term is the predicted `1 + 2·sqrt(nu_c/nuMax)`; the exact form matters
at low cap ratios (0.2346 vs 0.2000 at nuMax/nu_c = 1e2). Both are logged.

**The two axes are coupled in opposing directions, and that is the headline.**
Resolving a cap defect of relative size `eps` needs `N > 1/(xi·sqrt(nu_c/nuMax))`
— 50, 158, 500, 1581 cells at ratios 1e2…1e5 — so a *larger* cap needs a
*finer* mesh to see its own (smaller) defect. Meanwhile convergence cost grows
roughly as `N^2 · nuMax`: at nuMax/nu_c = 1e4, N = 40 needs ~180k iterations
for the identity to close, and the 1e5 column does not converge at any mesh
here. Measurability pushes toward fine meshes and high caps; convergence cost
forbids exactly that corner. Only 5 of 12 grid points converged, all with
nuMax/nu_c ≤ 1e3 — so the plug-width exponent could be verified at ONE cap
ratio, not swept. Reporting the exponent as "verified" from a sweep would
require a mesh nobody runs.

The stiffening is NOT a linear-solver artefact: `smoothSolver/symGaussSeidel`
and `PBiCGStab/DILU` stall at bit-identical residuals, and relaxation 1.0
diverges. It is the nonlinear nu↔gammadot fixed point, which contracts
algebraically. Two consequences the harness had to absorb: runs start from the
analytic Casson profile (worth orders of magnitude in iterations), and
**convergence is gated on the conservation identity rather than the residual**
— residual and profile drift are *change* measures, and a slowly-contracting
fixed point looks converged by both long before it is (residual 2.2e-9 with
the identity still at 2.5e-3). The identity compares two quantities that must
agree at the fixed point, so it measures distance to the solution, not step
size.

**Why this travels beyond the repo:** `nuMax` is an unreported degree of
freedom in published Casson haemodynamics. Rheology parameters get reported;
the regularisation cap does not. Since the cap sets the plug width, plug-region
results are not reproducible from the stated method — and the coupling above
means the cap cannot simply be raised to make the problem go away.

Two plug-width definitions are logged because they disagree, informatively.
Cap-active (where nu == nuMax) is mechanism-direct and reads the solver's own
viscosity field. Profile-flatness (|u − u_centre|/u_max < tol) is
threshold-dependent and answers a different question: at ratio 1e2 it reports
a *narrower* plug (the profile is measurably curved inside the cap region),
while at high ratios it reports a *wider* one (the creep is too small for the
threshold to see). Neither is wrong; they measure different things, and only
the first tracks the regularisation.

Note on xi: this case uses xi = tau_y/(G·h) = 0.2 to make the plug measurable.
Blood in a large artery has xi ~ 0.001–0.005 — the plug is under 1% of the
lumen, which is exactly why Casson is indistinguishable from Newtonian in the
aorta at peak systole.

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
All case types share one box mesh — one cell thick in z (`empty` front/back),
cyclic streamwise patches, separate bottomWall/topWall patches — and differ
only in their driving mechanism, split out per case type inside the runner:
`channel` (walls at ±h, `meanVelocityForce` fvConstraint, no entrance-length
effects) and `couette` (fixed wall at y=0, `fixedValue` moving wall at y=H,
no force). Each setup declares its extra templates (fvConstraints exists only
for channel), its Re→nu mapping, its u_ref, and its provenance
(meta.pressure_gradient exists only where a mean force does — cases opt out
of force-specific checks by construction). Profiles are sampled with
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
