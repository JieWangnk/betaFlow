# betaflow

A solver-independent validation framework for haemodynamic CFD. Each case pairs an
analytic reference with a solver run and asserts an explicit error norm against an
explicit tolerance, logging a provenance-stamped record. It is a regression
suite, not a one-off check: rerun it after any solver upgrade, scheme change,
or new boundary-condition library, and diff the committed results.

## Finding your way around

**I want to…**

| Goal | Do this |
|---|---|
| Run the fast checks (no solver install) | `python3 -m pytest -m analytic -v` — 27 tests, ~2 s |
| Run everything my machine supports | `python3 -m pytest tests/ -v` (default, needs OpenFOAM 14) — see [Running](#running) |
| Run one case, including a slow one | `python3 -m pytest tests/test_mc_channel.py -q -m ""` (the `-m ""` also selects `@slow`) |
| See what a case tests and why | Read its YAML in `betaflow/cases/` — each carries its citations, tolerances, and their reasons |
| Look up a measured number | `results/*.json` — see [Inspecting a results record](#inspecting-a-results-record) below |
| Understand the design and where every case sits | `docs/DESIGN.md` (architecture, case taxonomy table, correction policy) and [The hierarchy](#the-hierarchy) |
| Check an exact solution's own self-tests | `python3 -c "from betaflow.analytic import channel_impulse as m; print(m.verify_limits())"` — every module in `betaflow/analytic/` has `verify_limits()` |
| Inspect a generated OpenFOAM case | `_runs/<case>/` after any run (gitignored, kept for inspection) |
| Inspect the OpenLB app | `openlb_cases/mcChannel3d/` (C++ source with the findings as comments) + `betaflow/runners/openlb.py` |
| Point the instruments at production output | `tools/identity_check.py`, `tools/wall_traction_compare.py` — read-only by design |
| Re-run a solver-free audit | `python3 tools/hofmann_validity_audit.py`, `tools/eigentime_pe_sweep.py`, `tools/mc_channel_benchmark.py` |
| Add a case or runner | [Adding a case](#adding-a-case), [Adding a runner](#adding-a-runner) |
| Trace a correction or withdrawn claim | `git log` plus the `WITHDRAWN`/correction fields kept inside the records and docstrings |

**Contents, grouped by what each section is for**

*Orientation*
- [What "validated" means here](#what-validated-means-here)
- [The hierarchy](#the-hierarchy) — the replicate / re-examine / benchmark ladder
- [Layout](#layout) · [Running](#running)

*The fluid ladder (Tier 1, OpenFOAM)*
- [Current cases](#current-cases) — Poiseuille, Couette, refinement, Stage A
- [When error-accumulation reasoning applies — and when it does not](#when-error-accumulation-reasoning-applies--and-when-it-does-not)
- [Pipe geometry: what the framework assumed](#pipe-geometry-what-the-framework-assumed) — seven quietly channel-specific assumptions

*Rheology (yield stress and shear thinning)*
- [casson_steady](#casson_steady-a-non-physical-parameter-that-changes-the-answer) · [carreau_steady](#carreau_steady-the-contrast-case) · [womersley_carreau](#womersley_carreau-verification-without-an-analytic-reference)

*Particle transport (three runners, one physics)*
- [langevin_free](#langevin_free-the-first-non-openfoam-runner) · [taylor_aris](#taylor_aris-the-last-exact-analytic-reference) · [openfoam_particles](#openfoam_particles-the-three-way-check)

*The Eulerian scalar and LBM ladder*
- [The Eulerian scalar ladder](#the-eulerian-scalar-ladder-advection-diffusion-the-schemes-own-error-and-lbm) — advection_diffusion, moments, numerical_diffusion, lattice_boltzmann, the lbm runner, OpenLB first contact

*The comms benchmark (Tier 2)*
- [mc_channel](#mc_channel-the-comms-case-and-the-first-tier-2-benchmark) — three solvers against one exact CIR; the two-regime tail; the crossover-clock sweep

*Production and contribution*
- [Conservation check on production code](#conservation-check-on-production-code) — the read-only audit of real patient runs
- [Adding a case](#adding-a-case) · [Adding a runner](#adding-a-runner) · [OpenFOAM runner notes](#openfoam-runner-notes)

## Inspecting a results record

Every quoted number in this README traces to a JSON record in `results/`,
committed alongside the code. The naming rule: case `x.yaml` ↔ test
`tests/test_x.py` ↔ record `results/x.json` (variants get suffixes, e.g.
`mc_channel_openfoam.json`; solver-free audits are written by their
`tools/` script instead of a test). To read one:

    python3 -m json.tool results/mc_channel_benchmark.json | less

What the fields mean, in every record:

- **the numbers** — whatever the case measured, with names that say which
  quantity and which units;
- **`meta`** — how the run was configured (solver version, mesh or particle
  counts, timestep choices and what constrained them);
- **`git_sha`** — the 12-character commit the code was at, suffixed
  `-dirty` if anything tracked differed from it (regenerated outputs in
  `results/` and `report/` do not count as dirt);
- **`timestamp`** — UTC, seconds precision;
- **correction fields** — a claim that was withdrawn stays in the record
  under a `WITHDRAWN` name next to its replacement, with the reason. The
  repository treats the trail of corrections as data, so records are
  corrected in place, never silently swapped.

CI re-runs the suite and compares every record numerically against the
committed one (`tools/compare_results.py`, rtol 1e-3 / atol 2e-7, both
measured); `timestamp`/`git_sha` churn is excluded, anything larger fails
the build as a stale record.

## What "validated" means here

A case is validated when, for a stated Reynolds number and mesh level,

    metric(numerical profile, analytic reference) < tol

with the metric, tolerance, and non-dimensionalisation all declared in the
case YAML — never inside a runner. The committed `results/*.json` records the
error, mesh level, cell count, solver version, git SHA, and timestamp, so any
future regression is attributable to a specific change. A failing case is
diagnosed (analytic reference vs mesh vs boundary conditions), never "fixed" by loosening
the tolerance.

Two kinds of verification, deliberately distinguished (reviewers notice):

- **Code verification** — the case has an exact solution. The framework measures
  the *true* discretisation error and runs an **order-of-accuracy test**:
  observed p = log2(e_coarse/e_fine) must match the formal order of the
  scheme. This is stronger than any error-estimation procedure. All analytic
  rungs (Poiseuille, later Womersley) get this treatment;
  see `report/order_of_accuracy.md`.
- **Solution verification** — no exact solution (patient geometry, anything
  genuinely multi-dimensional). There the discretisation error can only be
  *estimated*, and GCI enters. GCI is never used where an analytic reference exists.

**Correction (superseded claim).** An earlier version of this file listed
Carreau rheology as solution-verification-only, "no exact solution". That was
wrong. In steady fully-developed 1-D channel flow the momentum equation
integrates once to

    tau(y) = G y        EXACTLY, for ANY rheology

— the same force balance that makes tau_w rheology-independent. So a
machine-precision analytic reference exists for *every* generalised Newtonian model: solve
the scalar monotone equation nu(gammadot)·gammadot = G y pointwise, then
integrate gammadot to get u. No ODE solve, no shooting, no GCI. `carreau_steady`
is therefore code verification with an order-of-accuracy test, and its analytic reference
self-verifies against the Newtonian and power-law limits to 1.8e-16 and
7.2e-16 before being used as ground truth. The same construction would cover
Herschel-Bulkley, Cross, and power-law without new machinery.

Non-dimensionalisation is where validation frameworks silently rot. The
Reynolds-number definition (bulk velocity, full channel height:
`Re = u_mean * 2h / nu`) is stated once in the analytic reference docstring
(`betaflow/analytic/poiseuille.py`), echoed in the case YAML, and
cross-checked by the test against the viscosity the runner actually used.

## The hierarchy

The project's shape is a three-tier ladder, stated in full in
`docs/DESIGN.md` (architecture, case taxonomy, correction policy, declared
scope). Nothing on a higher tier is trusted until the tiers below are green.

    Tier 0  RE-EXAMINE   analytic references re-derive published claims,
                         solver-free (self-checks; published-error findings)
    Tier 1  REPLICATE    one solver at a time against the exact solution
                         (null rung first, physics rung second)
    Tier 2  BENCHMARK    solver vs solver, exact solution as referee, each
                         solver's error predicted by its own reference
                         before it runs

How the sections below map onto the tiers: every per-case section is Tier 1
for its runner; the self-check counts quoted inside them are Tier 0; the
sections where one case runs under several runners (`langevin_free` and
`taylor_aris` under both `langevin` and `openfoam_particles`, and the
`mc_channel` comms case) are Tier 2. "Conservation check on production code"
applies the Tier-1 instruments to uncontrolled production output, which is
why it needed its own null test first.

## Layout

    betaflow/
      analytic/   analytic references — pure functions, no solver knowledge
      cases/      YAML case definitions (geometry, Re, analytic reference path, metric+tol)
      runners/    solver adapters — the ONLY layer that knows a solver exists
                  (openfoam, openfoam_particles, langevin, moments, lbm, openlb)
      metrics/    error norms over plain arrays
    tests/        pytest, one test per case
    results/      logged JSON records, committed
    tools/        standalone, READ-ONLY, imports nothing from betaflow —
                  so they can be pointed at production case directories

The design constraint: nothing above `runners/` may know OpenFOAM exists.

    run_case(case, runner="openfoam", n_cells=80) -> {..., "meta": {...}}

Metrics and tests consume only that dict. What is REQUIRED of a runner is a
`meta` sub-dict and arrays the case's declared metrics can consume — nothing
more. Fluid cases conventionally return `{"y", "u", "u_ref"}`, where `u_ref`
is the velocity the analytic reference normalises by, but that is a convention of those
cases and not of the framework: `runners/langevin.py` returns
`{"t", "msd", "msd_components", "D_expected"}` and plugged in with no change
above this layer. `meta` carries provenance (solver version, cell counts,
viscosity) and is never used for physics.

`tools/` sits outside the package on purpose. `identity_check.py` and
`wall_traction_compare.py` run no solver and write nothing into the case
directory they read, which is what makes them safe to point at production
output; `foam_mesh.py` therefore duplicates some parsing the runner also does,
and that duplication is the price of the isolation.

## Running

    python3 -m pytest -m analytic -v   # analytic tier, NO solver needed (27 tests, ~2 s)
    python3 -m pytest tests/ -v      # default: skips @slow studies (40 tests)
    python3 -m pytest -m "" -v       # everything (49 tests; > 45 min, not re-timed)
    python3 -m pytest -m slow -v     # the 9 slow studies alone

Tiering is by pytest marker (`addopts = -m 'not slow'` in pyproject.toml) and
wired into `.github/workflows/ci.yml`: the analytic tier gates every push with no
solver install, the default tier runs on push, the full tier runs nightly. Nine
tests are marked slow — the casson two-axis grid, womersley_carreau,
taylor_aris, the langevin error distribution, the three `openfoam_particles`
tests, and the OpenLB mc_channel run (skipped cleanly when the OpenLB build
is absent). `langevin_free` needs no solver and runs in ~5 s under the `langevin`
runner; the same case under `openfoam_particles` needs OpenFOAM 14 and
`libbrownianTracerCloud.so`.

The runner sources `/opt/openfoam14/etc/bashrc` itself; set
`BETAFLOW_OPENFOAM_BASHRC` to point elsewhere (the templates use OpenFOAM 14
Foundation syntax, so older versions will fail at dictionary parse).

Generated OpenFOAM cases land in `_runs/` (gitignored) for inspection.

## Current cases

| case | metric | analytic reference | mesh level | error | tol |
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
| womersley_carreau | identity_max_rel | tau_w(t) = h(G - d<u>/dt) — NO profile analytic reference | N=336, nt=1024 | 2.7e-7 | 1e-6 |
| carreau_steady | L2_velocity | Carreau-Yasuda channel (rootfind + quadrature) | N=160, Cu=10 | 6.1e-5 | 1e-2 |
| carreau_steady | wss_relative | tau_w = G h (rheology-independent) | N=160, Cu=10 | 0.0 | 1e-6 |

womersley_pulsatile's mesh level is CELLS PER STOKES LAYER (mesh refines
with alpha): a fixed-mesh alpha-sweep cannot distinguish high-alpha error
amplification from under-resolution. Its convergence notion is
cycle-to-cycle periodicity (tol 1e-6, cap 10 cycles, logged per cycle), not
residual-to-steady; runs start from the analytic t=0 profile, without which
the O(1) startup transient decays on the (2/pi^3) alpha^2 T homogeneous-mode
timescale and no cap is meaningful. The plane-channel analytic reference kernel is the
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
| tau_w = G_disc·h to round-off, rheology-independent | **confirmed** — exactly 0 at two grid points and ≤3.8e-11 on the four-point asserted subset (2.1e-7 at the fifth converged point) |
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
for the identity to close (20k leaves it at 1.8e-4), and the 1e5 column does
not converge at any mesh here. Measurability pushes toward fine meshes and high caps; convergence cost
forbids exactly that corner. Only 6 of 13 grid points converged, all with
nuMax/nu_c ≤ 1e3 — so the plug-width exponent could be verified at ONE cap
ratio, not swept. Reporting the exponent as "verified" from a sweep would
require a mesh nobody runs.

The stiffening is NOT a linear-solver artefact: `smoothSolver/symGaussSeidel`
and `PBiCGStab/DILU` stall at bit-identical residuals, and relaxation 1.0
diverges. It is the nonlinear nu↔gammadot fixed point, which contracts
algebraically. Two consequences the framework had to absorb: runs start from the
analytic Casson profile (worth orders of magnitude in iterations), and
**convergence is gated on the conservation identity rather than the residual**
— residual and profile drift are *change* measures, and a slowly-contracting
fixed point looks converged by both long before it is (residual 2.2e-9 with
the identity still at 2.5e-3). The identity compares two quantities that must
agree at the fixed point, so it measures distance to the solution, not step
size.

**Why this travels beyond the repo — WITH the prior work named.** The
mechanism is NOT novel: convergence of regularised solutions to the exact
viscoplastic model, the order of the error for different regularisations, and
the flows where regularisation is worst are treated in Frigaard, I.A. &
Nouar, C. (2005), "On the usage of viscosity regularisation methods for
visco-plastic fluid flow computation", J. Non-Newtonian Fluid Mech.
127:1-26 — whose framing is that these methods are popular and generally used
in an ad hoc manner. What this case adds is a measured instance in a
haemodynamics setting, with the mesh/cap coupling quantified.

The claim that might survive is about REPORTING, not physics, and it was
tested rather than asserted (`results/gapmap_numax_claim.json`, a Crossref +
PubMed corpus of 1906 records). The two literatures are structurally
disconnected: haemodynamics x regularisation co-occurs 4 times against 40.7
expected (ratio **0.10**, p = 6e-18), while the control pair haemodynamics x
wall-shear-stress is ENRICHED at 1.96 — so the method detects co-occurrence
where it exists.

**What that does and does not establish.** It supports a DISCONNECTION claim.
It does NOT establish a reporting rate: the counts are title+abstract only, a
regularisation cap is a methods detail that rarely reaches an abstract, and
the 2-of-140 figure is a visibility rate, not a reporting rate. The reporting
question needs full text. A direct precedent also surfaced and must be read
before any novelty claim is made — a hyperbolic-regularised Casson model for
pulsatile blood flow in a rigid artery sits squarely in the intersection.

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

## carreau_steady: the contrast case

Carreau has no regularisation parameter — nu is smooth and bounded between
nu_inf and nu_0 at every strain rate — so no cap, no plug, no second axis.
Committed predictions and outcomes (`results/carreau_steady.json`):

| prediction | outcome |
|---|---|
| tau_w = G_disc·h to round-off | **confirmed** — identity exactly **0.0** on most runs, ≤2e-12 on all |
| p ≈ 2 and NO floor | **confirmed** — p = 1.979, 1.990; error 9.50e-4 → 2.41e-4 → 6.07e-5, still falling |
| limits recovered | **confirmed** — analytic reference self-verifies to 1.8e-16 (Newtonian) and 7.2e-16 (power law) |
| no stiffening | **partly wrong** — geometric, but cost grows as ~Cu^0.6 (below) |
| flatness monotone in Cu | **wrong, and the analytic reference agrees** (below) |
| error coefficient grows with Cu | **wrong, same cause** (below) |

**Cost is not Cu-independent.** Iterations to a 1e-10 residual at N=80:
755 / 898 / 3916 / 15344 for Cu = 0.1 / 1 / 10 / 100, against **764** for the
Newtonian channel on the same mesh. So at Cu → 0 it matches Newtonian
*exactly*, then grows as ≈900·Cu^0.6. The mechanism claim survives even though
the number does: contraction stays **geometric** (measured rate 0.9948 per
iteration, a constant ratio) and the cost ratio versus Newtonian is bounded
(6.2×, 5.1×, 4.3× at N = 40/80/160 — falling with mesh). Casson, by contrast,
contracts **algebraically**, costs ~350× Newtonian, and diverges without bound
in nuMax. The unifying quantity is the viscosity contrast across the channel:
Carreau bounds it at nu0/nuInf by physics, casson lets nuMax push it anywhere.

**Two predictions failed for one shared reason, and the exact analytic reference confirms
it is physics, not a solver artefact.** Centreline flatness u_centre/u_mean
over the sweep is 1.4978 → 1.4635 → 1.3450 → **1.3875**: non-monotone, turning
back up at Cu = 100. The analytic reference gives 1.4996 / 1.4655 / 1.3470 / 1.3898 —
the same turn, to within the N=40 discretisation error. With a **finite**
viscosity ratio the near-wall fluid enters the *second* Newtonian plateau at
high Cu, so the profile de-flattens toward a parabola again. Set
nu_inf = 0 and monotonicity is restored exactly, converging to the power-law
value (2n+1)/(n+1) = 1.3333 — logged as `power_law_limit_flatness`. The L2
error follows the same curve (4.19e-4 / 5.19e-4 / 9.50e-4 / **8.39e-4**),
peaking precisely where the profile is furthest from parabolic. So the driver
is *departure from parabolic*, not Cu; the tests now assert that relationship
and that the solver reproduces the exact curve, monotone or not.

**Blood point** (Cho & Kensey, kinematic at rho = 1060: nu_0 = 5.283e-5,
nu_inf = 3.255e-6 m²/s, k = 3.313 s, n = 0.3568; h = 10 mm, u_mean = 0.3 m/s):
**Cu = 27.4**, flatness 1.4213 against the exact 1.4238. Note it thins *less*
than the n = 0.5 sweep point at Cu = 10 despite the lower power-law index,
because its viscosity ratio (0.0616) is six times larger — the same finite-plateau
effect, at physiological parameters.

## womersley_carreau: verification without an analytic reference

The first case with NO exact solution — the point of it. The unsteady term
breaks the force balance every previous analytic reference rested on, and nu(gammadot)
kills superposition, so the profile has no closed form. Patient geometry has
no analytic reference either; this is the rehearsal for that.

**Verified EXACTLY** (measured, not estimated):

| quantity | result |
|---|---|
| momentum identity tau_w(t) = h(G(t) - d<u>/dt), per timestep | 2.72e-07 |
| half-wave symmetry u(y,t+T/2) = -u(y,t) | 12.1x the periodicity residual (see below) |
| Cu -> 0 vs the exact Womersley cosh analytic reference | L2 amp 1.02e-03, phase 4.17e-04 rad |
| alpha -> 0 vs the exact steady Carreau analytic reference | L2 1.99e-03 |

**ESTIMATED by GCI** (ASME V&V 20, the first in this repo):

| functional | p | GCI (fine) | Fs | asymptotic |
|---|---|---|---|---|
| peak first-harmonic amplitude | 3.55 | 8.76e-05 | 3.0 | NO |
| wall-shear amplitude | 2.58 | 2.88e-04 | 3.0 | NO |

Neither sequence is in the asymptotic range, so the safety factor is raised
from 1.25 to 3 and that is reported rather than a tight band being quoted on
an unconverged sequence. The cause is understood: the stability limit below
forces n_t ~ N^2, so the temporal error falls as dy^4 while the spatial error
falls as dy^2, and a mixed A dy^2 + B dy^4 error gives an apparent order
between 2 and 4. Both sequences are monotone and the three solutions differ by
under 0.1% across a 4x mesh range.

**A stability limit with no Newtonian analogue.** The explicit deviatoric term
div(nu (grad u)^T) vanishes identically for constant nu, so Newtonian runs
never feel it. Under a variable viscosity it does not, and it imposes
nu0*dt/dy^2 = O(1): measured stable at Fourier 1.5 and 3.3, divergent at 13.2.
So **n_t must scale as N^2, not N**. Worse, the failure is not always a crash —
at Fourier 279 a run completed normally while producing garbage (the two wall
shears disagreed, -1.06 vs +3.27, and <u> oscillated violently). Nothing in
the solver output flagged it; only the identity did, opening to 3.18. That is
why the identity is the transient convergence gate.

**Mesh sizing must use nu_wall, not nu0.** Shear thinning gives
nu_wall/nu0 = 0.117 here, so the Stokes layer is thinner and the EFFECTIVE
Womersley number is alpha_eff = 29.2 against a nominal 10. Sizing on nu0 would
have given N=112 instead of N=336 and looked like a discretisation problem
rather than a sizing mistake.

**Odd-harmonic content** is the signature of the nonlinearity and the analogue
of the phase metric in womersley: half-wave symmetry admits only odd
harmonics, and a linear response has none above the first, so A3/A1 measures
the constitutive nonlinearity directly. Measured 0.0066 (velocity) and 0.0186
(tau_w) at Cu = 10, collapsing to 3.95e-06 at Cu = 0.01 — a factor of 4700.

**Periodicity costs 10x more than the linear case.** The transient decays at
0.865/cycle, set by nu0 (the NOMINAL alpha) rather than nu_wall, because the
slow mode lives in the low-shear core. Reaching a 1e-6 periodicity residual
would take ~90 cycles (~30 min at the finest level), so the budget is capped
at 20 and the ACHIEVED periodicity is reported: cycles_to_periodic is None at
every level. Half-wave symmetry is then bounded against it — the ratio is 12.1
here and was 11.7/12.6/12.4/12.2 at 4/8/12/16 cycles, so the symmetry residual
is incomplete periodicity and not an independent violation.

In the alpha -> 0 limit the first-harmonic amplitude is 0.81 of the steady
peak. That is not an error: a shear-thinning fluid driven by G cos(wt)
responds non-sinusoidally, so the fundamental carries less than the full
amplitude. It is the A3/A1 nonlinearity seen from the other side, and the
profile SHAPE still matches the steady analytic reference to 2e-3.

## Pipe geometry: what the framework assumed

Three cases were ported to a circular pipe on an axisymmetric wedge —
`pipe_poiseuille_steady`, `pipe_casson_steady`, `pipe_womersley_pulsatile` —
to answer one question: do the abstractions hold for any geometry, or are they
quietly channel-specific? **Quietly channel-specific, in seven places.** All seven are
now per-case-type rather than constants.

| assumption | where | consequence |
|---|---|---|
| box `blockMeshDict` and `0/p` were "shared" | `_SHARED_FILES` | geometry files can't be shared; now `_BOX_FILES` / `_WEDGE_FILES` |
| wall patches are the pair `bottomWall`/`topWall` | tau parser | a pipe has ONE wall; patch list now per type |
| transverse extent is `2h` | mesh sizing from cells-per-Stokes-layer | pipe over-resolved 2x, which raised the viscous Fourier number to 6.2 and **diverged** — a sizing bug presenting as a stability failure |
| momentum lever arm is `h` | transient tau reference and identity | pipe's is `a/2`; identity read exactly **1.0** (100% error) |
| front/back are `empty` | field templates | wedge is a collapsed real direction, not an ignored one |
| sample line at a fraction of the domain thickness | `z_mid` | must be the wedge symmetry plane, z = 0 |
| transient file set is a box | `_WOMERSLEY_FILES` | now split by geometry |

The driving abstraction (`meanVelocityForce`, cyclic streamwise) and every
metric needed **no change at all**, and the analytic references transferred as pure factor
changes from tau(y) = G y to tau(r) = G r / 2.

**The axis is not a patch.** r = 0 is a coordinate singularity; the block is
COLLAPSED onto it by repeating the axis vertices in the `hex` entry, so the
axis-adjacent cells are prisms and the would-be axis faces have zero area and
belong to no patch. `checkMesh` confirms: 156 hexahedra + 4 prisms, "2
geometric (non-empty/wedge) directions". Declaring an axis patch is the
common error.

**Wedge faceting is a refinement-independent bias, and only the identity
caught it.** A wedge's outer boundary is a flat chord, not an arc. With
vertices on the circle r = a the discrete volume-to-area ratio is
a·cos(theta)/2, so the exact discrete force balance gives
tau_w = G a cos(theta)/2 — a relative bias of 1 − cos(theta) = 9.52e-4 at
theta = 2.5 deg. Measured before the fix: **9.518e-04**, against a predicted
9.5178e-04. It does NOT shrink with radial refinement, so no
mesh-convergence study would ever reveal it; it would sit as a permanent
~0.1% WSS offset. Placing the vertices at a/cos(theta) puts the chord
midpoint on the circle, restoring V/A = a/2 exactly — identity 9.518e-04 →
**2.5e-12**.

**J0 belongs to the pipe.** An earlier prompt in this project specified
"Bessel functions of complex argument" for a PLANE CHANNEL; that was wrong and
was overridden at the time in favour of the complex cosh. Both kernels now
exist, each labelled with its geometry: `betaflow/analytic/pipe.py` (J0,
Womersley 1955) and `betaflow/analytic/womersley.py` (cosh). They must never
be swapped — and the uncomfortable part is how hard a swap is to detect:

| alpha | amplitude misfit | phase misfit [rad] | WSS amplitude misfit |
|---|---|---|---|
| 2 | 1.69e-1 | 3.89e-1 | 3.39e-1 |
| 5 | 6.25e-2 | 5.22e-2 | 4.64e-1 |
| **10** | **9.14e-3** | **1.46e-2** | **4.82e-1** |
| 20 | 2.59e-3 | 4.56e-3 | 4.91e-1 |

(These rows are the 201-station evaluation the test uses. The
station-CONVERGED metric values — 9.158e-3 amplitude, 1.459e-2 rad phase at
alpha = 10 — are in `results/kernel_misfit_operating_point.json`, with the
convergence sweep; the manuscript quotes those.)

The mechanism: profiles converge as alpha rises because both geometries
develop the same structure — an inertia-dominated plug core, which is
geometry-blind, plus a Stokes layer whose thickness sqrt(2 nu/omega) is set by
frequency rather than curvature and which is locally flat at any wall. So
SHAPE stops discriminating. Wall shear does not converge, because it carries
the lever arm relating wall traction to the driving force: h for a channel,
a/2 for a pipe. That factor is pure geometry and never washes out.

Stated generally, and this is now the third instance after 0d's silent failure
and Euler's amplitude preference: **the discriminating quantity is the one
tied to the geometry by a conservation relation, not the one that looks most
informative. Profile agreement is weak evidence.**

At the case's alpha = 10 the wrong kernel misfits the profile by less than
this case's OWN tolerances (1e-2 amplitude, 2e-2 phase) — a profile-only
validation would pass a swapped kernel — while wall shear misfits by 48%,
because tanh(K)/K and 2·J1(b)/(b·J0(b)) differ by a factor approaching
exactly 2 at large alpha. The profile metrics get WORSE at discriminating as
alpha rises (both kernels tend to a flat core plus a thin Stokes layer) while
WSS gets better. The test therefore asserts on WSS and records the amplitude
blind spot explicitly.

Results (wedge topology, `results/pipe.json`): Poiseuille p = **1.972,
1.982** with identity ≤ 5e-12; Casson identity **8.55e-11** with the realised
plug ratio xi_c = **0.200008** against a target of 0.2; Womersley L2
amplitude **9.32e-4**, phase **4.16e-4 rad**, WSS amplitude **9.38e-5**,
identity **1.19e-10**.

**The fix generalises, and the choice of mesh follows the purpose.**
Circumscribe rather than inscribe: a regular n-gon with INRADIUS a has area
n·a²·tan(pi/n) and perimeter 2n·a·tan(pi/n), so V/A = a/2 **exactly at any n**
— the tan cancels identically. That removes the faceting bias at any
circumferential resolution rather than mitigating it. But there is no free
lunch, and the three choices optimise different things: circumscribing zeroes
V/A while over-predicting cross-sectional area by +0.051% at n=80; inscribing
under-predicts area by −0.103%; equal-area gets area exactly and V/A neither.
**Use V/A for the identity gate, area for particle concentration and wall
position.** Those are different meshes, and pretending one serves both is how
the error gets hidden.

**Why p ≈ 2 survived the faceting bias — and where it would not.** The wedge's
uniform geometric scaling is absorbed by `meanVelocityForce`: G_disc
self-adjusts to hit the target mean velocity, so a cos(theta) error in the
effective source cancels against the analytic profile evaluated at G_disc.
This is the same self-consistency that made Poiseuille WSS second-order.
**That immunity is a property of FORCE-DRIVEN flow, not of the mesh.** A
fixed-pressure-drop or fixed-inlet-flow setup — which is what patient-specific
cases actually use — has no G_disc to absorb it, and the bias appears directly
in the profile. Do not read "we fixed the faceting bias" as fixed for
everyone.

**Architecture: wedge for fluid verification, O-grid for particles only.** A
corrected wedge is EXACT for an axisymmetric solution — there is no
circumferential discretisation of the physics at all — whereas the O-grid
necessarily carries a geometric floor that radial refinement never touches.
Running the three-case re-verification on the O-grid would measure that floor,
not agreement. Committed prediction if it is ever run: **O-grid and wedge
disagree at ~1e-3 on the profile at n_circ = 80, and the gap does not shrink
under radial refinement** — the same signature as casson's nuMax floor, a
different cause. This is not deferred work; it is the right split.

**O-grid: the same faceting law, confirmed on a second topology.** A 5-block
3-D O-grid (8000 hexahedra, `tools/ogrid_blockmesh.py`, `checkMesh` clean, max
skewness 0.98) was built and pipe Poiseuille run on it. Identity gap:
**7.710e-04**, against a predicted 1 − cos(pi/80) = **7.7096e-04** for its 80
circumferential faces. So the wedge result was not a wedge quirk — it is the
general polygonal law, a boundary polygon with vertices ON the circle having
area/perimeter = R·cos(pi/n)/2. The topologies differ in one respect that
matters: the wedge half-angle is fixed by construction so its bias never
shrinks under refinement, while the O-grid's falls as 1/n_circ² — but only
under CIRCUMFERENTIAL refinement, which a radial convergence study never
performs. STATUS: the O-grid is a standalone tool, not yet wired into
`runners/`, and the 1/cos correction is applied only on the wedge path; the
full three-case re-verification on the O-grid is NOT done.

CITATIONS. Every pipe equation traces to a source recorded in both the analytic reference
docstring and the case YAML: Batchelor (1967 §4.2) and Sutera & Skalak (1993)
for Hagen-Poiseuille; Reynolds (1883) for the Re convention; Womersley (1955)
for the J0 kernel; Casson (1959) for the constitutive law and Fung (1997
ch. 3) for Casson blood flow in tubes. Two things are flagged as having NO
published anchor rather than being given a plausible one: the symbol xi_c is
attributed to Gentile et al. (2008) **on the user's authority, unverified
here** (the quantity r_p = 2·tau_y/G needs no citation — it follows from
tau(r_p) = tau_y and is checked by differentiation), and `nuMax` is an
OpenFOAM regularisation parameter, a numerical device with no physical source.

## langevin_free: the first non-OpenFOAM runner

Free Brownian motion of a 50 nm particle in plasma (T = 310 K,
mu = 3.5e-3 Pa·s), giving D = **1.2975e-12 m²/s** and an rms displacement of
2.77 um in 1 s — the expected order for a nanoparticle in plasma, water at
37 °C being ~5x less viscous. No flow, no walls, no CFD solver.

**The architecture result.** betaflow's central claim is that nothing above
`runners/` knows about the solver. Until this case every runner WAS OpenFOAM,
so the claim was asserted, never tested. Adding a pure-Python Langevin
integrator required **no change anywhere above the runner layer** — not to
`run_case`, not to the metrics, not to provenance, not to the case-loading in
tests. One thing was wrong and is now corrected: the documented return
contract said `{"y", "u", "u_ref", "meta"}`, which is fluid-specific. The
dispatch never enforced it, so a runner returning `{"t", "msd", ...}` plugged
in regardless. The layering held; only its statement was too narrow.

**The convergence axis is particle count, not resolution**, and the rate is
Monte Carlo:

| N | RMS slope error (8 replicas) |
|---|---|
| 1 000 | 2.48e-2 |
| 10 000 | 8.42e-3 |
| 100 000 | 2.43e-3 |

Measured exponent **0.504** against the expected 0.5 (ratios 2.94 and 3.47 per
decade versus sqrt(10) = 3.16). A refinement helper that assumed p ~= 2 would
be measuring the wrong thing entirely.

Predictions, all confirmed: MSD slope recovers 6 D t; error scales as
1/sqrt(N); **timestep independence** (8.3e-3, 3.6e-3, 4.1e-3 at dt = 0.02,
0.01, 0.005 — no trend, because Euler-Maruyama is EXACT for free diffusion:
the true increment is Gaussian with variance 2 D dt, which is precisely what
is sampled); and isotropy (per-component MSD spread 3.0e-2 against a
statistical scale sqrt(2/N) = 1.4e-2, i.e. ~2 sigma).

**The tolerance is now sigma-scaled, which is the half of the fix that
matters.** The original 2e-2 sat only ~2.4x the statistical error at
N = 10 000, so ~1 seed in 60 would have failed. Raising N to 100 000 fixes
that — and is the WEAKER half, because a fixed tolerance on a Monte Carlo
quantity always passes at large enough N and therefore verifies nothing about
the sampler. The assertion with content is scaled to the predicted error:

    |slope/(6D) - 1| < 4 * sigma(N)

which is N-independent in units of sigma (failure rate 6.3e-5 at any N) and
asks the question that has content — not "is the answer close" but **"is the
answer within its own error bar"**. A sampler with correlated draws inflates
the measured spread and fails this immediately while sailing through a fixed
2e-2 at N = 100 000. This is the same rule as the conservation gate: there,
the residual measured step size while the identity measured distance to the
fixed point; here, a fixed tolerance measures "is N big enough" while the
sigma-scaled one measures the statistical model.

**The distribution check then caught a wrong error MODEL — mine.** Running 20
seeds and testing the DISTRIBUTION of z (roughly 68% within 1 sigma, 95%
within 2) is the Monte Carlo analogue of an order-of-accuracy test: it tests
the error model rather than one draw from it. It immediately showed RMS z =
0.64, i.e. the real spread was ~35% below the assumed sigma. The assumed
sqrt(6)/3 = 0.8165 is the relative sd of the MSD at a SINGLE time; the metric
fits a slope through the origin over all times, whose MSD values are
correlated by Cov(|r(s)|², |r(t)|²) = 24 D² min(s,t)². Carrying the sums
through gives **sd(slope)/6D = sqrt(18)/6 / sqrt(N) = (1/sqrt(2))/sqrt(N)**,
confirmed three ways (discrete sum 0.707133, independent direct simulation
0.708732, continuum 0.707107) and against 60 seeds of the runner itself
(0.769 ± 9%, RMS z = 1.09).

**Three distinct sigma laws now exist in the framework and must not be
interchanged**: 0.8165/sqrt(N) for a single-time 3-D MSD, 0.7071/sqrt(N) for a
through-origin slope fit, and sqrt(2/N) for a variance estimator such as an
axial dispersion coefficient. Baking any one of them in as "the Monte Carlo
error law" is the statistical analogue of reusing a channel analytic reference on a pipe.

Fluctuation-dissipation is asserted directly: the friction coefficient in the
noise amplitude and in Stokes drag must be one and the same zeta = 6 pi mu a
(round-trip error 0.0). Taking them from independent constants is the classic
implementation bug, and it silently changes the MSD slope.

Overdamped (inertialess) Langevin is justified by St = 5.0e-9 at these
scales, reported in meta rather than assumed.

## taylor_aris: the last exact analytic reference

Dispersion of a nanoparticle in laminar pipe flow, on `runners/langevin.py`
with an analytic Poiseuille field — no CFD solver. After this there are none:
margination, capture efficiency and deposition have no closed forms, so this
is the final rung where a result can be checked against truth rather than
estimated.

The analytic reference self-verifies before use, at machine precision: Decuzzi Eq. 15
equals Taylor-Aris with Stokes-Einstein substituted (2.2e-16, since
6/48 = 1/8); Eq. 18 is the stationary point of Eq. 15 found by differentiating
it rather than hardcoded (2.2e-16); Pe at R_cr is exactly sqrt(48)
(4.4e-16); and E[u] = U, Var(u) = U²/3 by quadrature (2.5e-13, 1.0e-12).

**Four exact anchors, three new in kind.**

| anchor | result |
|---|---|
| D_eff = D + a²U²/(48D), from the long-time variance slope | z = 2.20, 0.09, 0.29 at N = 10⁴, 3×10⁴, 10⁵ |
| short time: sigma_x² − 2Dt → (U²/3)t², a DIFFERENT power of t | exponent **1.925**, prefactor 0.930 of U²/3 |
| P(r) = 2r/a² at all times — THE GATE | KS 0.0014–0.0049 across t/tau_r = 1.3 to 10 |
| the SHAPE of D_eff(R), via a two-parameter fit | **R\* = 50.36 nm vs Eq. 18's 50.00 nm (0.7%)** |

The radius sweep is the strongest of the four because it tests a shape rather
than a point: D_eff falls with particle size below R_cr and rises above it,
and R* comes from fitting D_eff(R) = A/R + BR to all nine points rather than
scanning for a minimum the curve is flat around. Individual points agree with
Eq. 15 to 1–3% at N = 10⁴.

**Two constraints, both anchored rather than asserted.** The duration is
t_max = 10 tau_r, not 50: radial relaxation decays as exp(−beta_1² t/tau_r)
with beta_1 = 3.8317 the first zero of J1, i.e. exp(−14.68 t/tau_r), which is
4e-7 at one tau_r. The asymptotic regime arrives at ~tau_r, so the fit window
[2, 10] tau_r is generous and 50 tau_r would be 5x wasted. That alone is the
difference between 10¹¹ and 10⁹ particle-steps.

The radial step ratio epsilon is then **calibrated by the gate rather than
chosen**: sweeping epsilon = 0.05 / 0.025 / 0.0125 gives KS/floor = 0.63,
0.86, 0.88 — at or below the statistical floor at every value, and NOT falling
with epsilon. So epsilon = 0.05 is verified, not assumed; had KS sat above
floor and fallen, a real wall-handling error would have been resolving. The
check that validates the physics also calibrates the timestep, with no
separate convergence study.

**The wall scheme is specular reflection with the exact crossing**, iterated.
Not rejection, which depletes or piles up the near-wall population, and not
radial folding r → 2a − r, which is measure-distorting in 2-D because the area
element 2·pi·r·dr differs between r and 2a − r. Either error would bias D_eff
invisibly through the velocity sampling — nothing in the dispersion number
itself reveals it, which is exactly why P(r) is the gate.

**One honest caveat on the error-rate fit.** The particle-count study uses one
seed per N, so the fitted "MC exponent" of 1.35 is not an estimate of the
1/sqrt(N) law — three single draws cannot measure a standard deviation. The
z-scores being O(1) at every N is the meaningful check, and the scaling law is
verified properly in `langevin_free` with 8 replicas per point.

**Physical status.** U ~ 3 um/s is far below any real vessel; it is chosen so
that R_cr = 50 nm sits at the centre of the sweep. R_cr scales as 1/(mu·a·U),
so at physiological velocities it is sub-nanometre and every real nanoparticle
sits on the convection-dominated branch — which IS the content of Decuzzi
Eq. 18. Making the minimum observable requires a slow flow. Same status as
xi = 0.2 in casson_steady: a verification setting, not physiology.

## The Eulerian scalar ladder: advection-diffusion, the scheme's own error, and LBM

Everything above transports PARTICLES. A lattice-Boltzmann or finite-volume
molecular-communications channel model transports a CONCENTRATION FIELD, and
nothing here tested that formulation. Five components close the gap, built in
the order analytic reference -> runner -> case, with the constants derived rather than
quoted at every step.

### advection_diffusion: the Eulerian twin (43 self-checks)

`betaflow/analytic/advection_diffusion.py` is the Eulerian twin of
`taylor_aris.py`: same physics, same exact answers, independent
implementation, cross-checked module against module. The Taylor-Aris
constants are DERIVED from the Aris cell problem with the pipe as a positive
control — the route must return Aris's published 1/48 before its channel
answer (2/105) is worth anything. Var(u)/U² = 1/3 (pipe) and 1/5 (channel).
Six anchors beyond D_eff, each catching something D_eff cannot: the variance
intercept (weights the transverse spectrum as beta^-8 where D_eff weights it
beta^-6, so a fitted D_eff cannot absorb it), the third cumulant (EXACTLY
blind to axial diffusion, and its sign differs between the geometries),
<u'^3> = 0 exactly for a pipe, the transverse-distribution gate, the
point-release centroid offset (a pulse released exactly on the u = U
streamline still ends up permanently BEHIND, by -1/96 and -1/90), and the
balance Peclet. A spectral route closes the same constants on Rayleigh sums
over zeros of J1 and on zeta(4), zeta(6), zeta(8) — arithmetic identities no
algebra slip can reproduce by luck.

CITATION TRAIL, recorded per source: the channel 2/105 is anchored to
Ajdari, Bontoux & Stone (2006), read in full; the Wooding (1960) attribution
was WITHDRAWN after checking (it is a Hele-Shaw stability paper); and Beard
(2001) PUBLISHED 33/560 for exactly this constant — three times the correct
value, corrected in print by Dorfman & Brenner — which is the best available
evidence the constant is easy to get wrong.

**Corrections along the way, kept in the record:** the docstring's
`asymptotic_onset` figure (0.63) was the value at tolerance 1e-4, not 1e-6
(0.941); a first version claimed the pipe "has no odd/even selection rule"
when the disc's true slowest Neumann mode is non-axisymmetric at
beta² = 3.390, 4.331x slower than the 14.682 the symmetric case uses —
`asymptotic_onset` now takes `symmetric_release` and returns 4.075/5.599
when the symmetry is broken.

### The moments runner: no axial mesh, on purpose

`betaflow/runners/moments.py` solves the Aris moment hierarchy on the
cross-section — the axial coordinate is integrated out EXACTLY, so the
runner measures the transverse operator to high precision and says NOTHING
about a solver's axial advection scheme. That scope is stated in the module,
the case (`scalar_dispersion`), and a test that fails if the claim is
silently outgrown. Time integration is by matrix exponential (no timestep
error), which is what lets the variance INTERCEPT be measured to 3e-5 and
shown to converge at second order (2.0, 2.001 pipe; 1.999, 2.0 channel) —
the convergence assertion lives on the intercept because D_eff is already at
1e-9 at the coarsest level and has no headroom left to measure an order
with. One instrument finding: the point-release offset came out 7e-3 in BOTH
geometries — two geometries agreeing to 1% is a common-cause signature — and
was the discrete mean velocity differing from the nominal U (the G_disc
lesson again); comparing against u_disc collapsed it to 2.5e-5.

### numerical_diffusion: the scheme's own error as the analytic reference (16 self-checks)

A solver with physical diffusivity D and numerical diffusivity D_num
produces a pulse IDENTICAL to the exact solution for D + D_num — no single
profile separates them. What separates them is that D_num depends on dx, Co
and the scheme while D depends on none of those, so the discriminating
experiment is a SWEEP. `betaflow/analytic/numerical_diffusion.py` gives
D_num = (u dx/2)(1 - Co) for explicit upwind, derived by von Neumann
analysis of the exact amplification factor and verified against a real 1-D
solver at D = 0: ratio 1.0000 at nine (N, Co) combinations. At Co = 1 the
scheme is a pure one-cell shift and EXACT (measured -1.9e-17); the -Co term
is entirely temporal truncation. The dispersive coefficient E3 is exported
for its SIGN STRUCTURE only (roots at Co = 1/2 and 1, both observed): its
measured magnitude sits ~8% above prediction and does not converge, which is
stated as unresolved rather than absorbed into a tolerance. The application
number: at u = 0.3 m/s, dx = 0.5 mm, D = 1e-9 m²/s the artefact fraction is
0.99999 — the scheme contributes five orders of magnitude more spreading
than the fluid, invisibly.

### lattice_boltzmann: constants derived, gaps declared, then measured (95 self-checks)

`betaflow/analytic/lattice_boltzmann.py` computes every lattice constant
from the velocity sets rather than tabulating it, and RAISES if table and
derivation disagree. The findings, in order of sharpness:

- **D = c_s²(tau - 1/2), and the -1/2 is a silent failure.** Dropping it
  gives an error factor tau/(tau - 1/2): 1.33x at tau = 2 but 51x at
  tau = 0.51 — a code calibrated at large tau fails by orders of magnitude
  exactly where a low-diffusivity solute lives. tau < 1/2 is NEGATIVE
  diffusivity: the stability boundary as a transport coefficient.
- **c_s² is a WEIGHT FAMILY, not a lattice-name constant.** D3Q7:
  c_s² = omega/3 with rest weight 1 - omega; the textbook omega = 3/4 gives
  1/4, omega = 1 gives 1/3 — both published values are right for different
  weights. The name-keying trap exists WITHIN OpenLB itself: its general
  D2Q5 descriptor is 1/3 while its thermal MRT D2Q5 declares 0.2. Confirmed
  at source level against an OpenLB 1.9 checkout: cs2<3,7> = {1,4} in
  src/descriptor/definition/common.h.
- **The reduced sets cannot carry momentum**: D2Q5 and D3Q7 fail
  fourth-moment isotropy, so a coupled simulation needs a full set for the
  fluid lattice whatever the scalar uses.
- **The Ma² law, RESOLVED by derivation**: the first-order-equilibrium BGK
  scheme (which OpenLB's ADE dynamics use, source-confirmed) realises
  D_eff = (c_s² - u²)(tau - 1/2) = D(1 - Ma²) — the coefficient is exactly
  -1. The second-order equilibrium cancels it at standard weights and
  OVERCORRECTS by +5u² on OpenLB's thermal D2Q5 weights. Derived by the
  same von Neumann route, then verified on an actual lattice at ratio
  1.000000000.
- **The ADE Dirichlet slip and the magic parameter**: anti-bounce-back
  walls leave a uniform offset vanishing only at
  Lambda = (tau - 1/2)² = 3/16 (arXiv:1603.09577, whose printed Eq. 73
  carries a SIGN ERROR — the corrected relation satisfies the magic
  condition identically, recorded alongside Beard 2001). The slip converges
  as 1/N², so it is NOT the wedge-bias analogue: it inflates the constant,
  not the order.

Still open, declared in `UNRESOLVED`: the Ma² law off-axis and in 3-D, and
the tau-dependent MOMENTUM bounce-back wall position (its one sourced claim
failed adversarial verification 0-3; He/Zou/Luo/Dembo 1997 and Ginzburg's
TRT papers remain unread).

### The lbm runner: the analytic reference measured on an actual lattice

`betaflow/runners/lbm.py` (pure numpy, D1Q3 + D2Q5(omega), both equilibrium
orders, periodic ring + anti-bounce-back walls with source) confronts every
analytic reference claim with a real collide-and-stream lattice — `tests/test_lbm.py`,
~5 s, default tier. The slip experiment BISECTS the measured slip and lands
the zero at tau = 0.9330127019 against the exact 1/2 + sqrt(3)/4 =
0.9330127019: Lambda = 3/16 to ten digits, MEASURED. Three convention
findings the measurement forced: the published prefactor is off by exactly 4
until the paper's N is read as the HALF-width; the simple source scheme
shifts the whole curve by exactly -S/2 (the missing half-step of the He-Luo
scalar redefinition); and at fixed lattice source the slip is N-independent,
which IS the published 1/N² law since Delta_phi grows as N².

### OpenLB first contact: the depletion law in the wild

OpenLB 1.9 was built from source and its shipped ADE benchmark
(advectionDiffusion1d, the Simonis-Frank-Krause 2020 setup) run as shipped:
N = 50, latticeU = 0.4, tau = 5, requested D = 1.5. Measured FROM ITS OWN
WRITTEN FIELDS (`tools/openlb_first_contact.py`, read-only;
`results/openlb_first_contact.json`):

    D_eff = 0.908   against a requested 1.5    (40% deficit)
    u_eff = 9.09    against a requested 10     (9% slow)

against the exact-eigenvalue prediction 0.9031 / 9.102 — agreement 0.5% and
0.1%. The converter itself is right (its tau = 5 satisfies
D = c_s²(tau - 1/2) exactly); the depletion is the SCHEME's. The example's
own error print decays reassuringly (0.161 -> 0.024) while the realised
transport coefficients sit 40% and 9% off — the reassuring-headline-next-to-
silently-wrong-physics structure again, in the target code's shipped
benchmark. No blame attaches to the published EOC study (diffusive scaling
puts this error inside the O(N^-2) budget); the trap is any FIXED-resolution
run at the shipped latticeU. One refinement flowed back into the analytic reference: at
large tau the k -> 0 law saturates slowly (0.78 vs the true 0.9031 even at
k = 0.123), so quote the eigenvalue at the actual wavenumber.

**Instrument failures during this strand, recorded where they happened:**
the modified-equation route gave E3 a single root where the measurement
showed two (the Fourier route needs no substitution and got both); a
constant 0.998667 measurement ratio was attributed to a plausible k⁴
truncation story until a pulse-width sweep refuted it — the cause was an
off-by-one, 749 elapsed steps divided by 750; and a 50-digit eigenvalue
check read d_eff ~ 9252 because weights passed as doubles sum to 1 - 5e-17,
which sits directly in the conserved eigenvalue. Naming the alternative was
not enough for the first of these; TESTING it was what caught the bug.

## openfoam_particles: the three-way check

`betaflow/runners/openfoam_particles.py` is the THIRD runner, and it is the
first time one analytic reference has been answered by two independent solvers. Until now
the layering claim rested on one non-OpenFOAM runner plugging in without
changes above `runners/`; now `langevin_free` and `taylor_aris` each run
through two of them, against the same analytic reference and the same metrics —
analytic reference vs `langevin` vs OpenFOAM 14's modular Lagrangian `brownianTracer`
cloud. A disagreement can be localised rather than merely detected.

**It closes.** Results in separate `*_openfoam.json` files, deliberately, so
the committed `langevin` records stay byte-stable and the CI diff gate still
means something.

| case | metric | error | sigma law | z |
|---|---|---|---|---|
| langevin_free | MSD slope vs 6·D·t | **0.24%** | 0.7071/sqrt(N), N=1e5 | 1.06 |
| taylor_aris | D_eff vs Eq. 15 | **0.33%** | (sqrt6/2)/sqrt(N), N=1e4 | 0.27 |
| taylor_aris | radial KS vs P(r)=2r/a² | 6.93e-3 | floor 1.36e-2 | **0.51x floor** |

**The stock force model does NOT reproduce Stokes-Einstein here.** OpenFOAM's
`BrownianMotionForce` measured D/D_SE = **0.38 to 0.59 on this machine,
varying with maxCo** — a diffusivity that depends on the timestep control is
not a diffusivity. The replacement works at displacement level and holds the
amplitude-dt / applied-dt identity by construction, which is the same move as
everywhere else in this repo: pin the quantity with an exact relation rather
than hope the discretisation preserves it. Caveat stated plainly: this is one
machine and one OF14 build, and no attempt has been made to trace the stock
model's factor to its source.

**The radial KS gate earned its keep, exactly as specified.** A plain rebound
wall — reflecting the resolved velocity but not the noise velocity — leaves an
effectively STICKY wall: KS **3.31x** its floor and D_eff **+8.3%**. The
+8.3% alone would have read as ordinary Monte Carlo scatter. What caught it is
that P(r) = 2r/a² is exact, independent of the dispersion physics, and
therefore able to discriminate; `brownianReboundVelocity` reflects the noise
velocity as well as U, and the KS drops to 0.51x floor. This is the third
recorded case of an exact side-relation catching what the headline number
could not (after the wedge faceting and the kernel swap).

**Scope, stated because it is narrower than the langevin coverage.** Free
diffusion and the long-time D_eff anchor with the radial KS gate. The
short-time t² ballistic anchor and the radius sweep for R* stay langevin-only.
Both tests are `@slow` and need OpenFOAM 14 plus
`libbrownianTracerCloud.so`; the runner resolves `FOAM_USER_LIBBIN` through
the OF14 bashrc, because fresh shells on this machine default to OF12. The
analytic tier is unaffected (14/14, no solver).

## mc_channel: the comms case, and the first Tier-2 benchmark

The molecular-communications channel impulse response (CIR): release
particles at t = 0, uniformly over a pipe's cross-section, and count the
fraction inside a transparent receiver window over time. Geometry and
Pe = 200 are Hofmann et al. 2024's Table 1 (doi:10.1109/ACCESS.2024.3438243);
the exact flow-dominated solution (their Eq. 13) was re-derived
independently via the uniform-speed lemma before use
(`betaflow/analytic/channel_impulse.py`, 18 self-checks). The dimensional
split V = 1.5 mm/s, D = 1.5e-9 m^2/s is ours and the case YAML says so.

**The analytic model's own limits, measured then corrected**
(`results/hofmann_validity_audit.json`, `eigentime_pe_sweep.json`): the
model's log-divergent tail describes at most 8.5 / 6.2 / 5.0 peak-times
past release at the three receivers. The first version of this claim
(27 / 6.8 / 3.4) converted the crossover through the radial relaxation
eigentime tau_r/beta_1^2, resting on a one-point agreement (predicted 6.8
vs measured 6.5 at the middle receiver) — and the pre-registered Peclet
sweep REFUTED that clock: over Pe 50-800 and all three receivers the
measured scaling is t_cross = K tau_r^0.31 dbar^0.73, the layer-escape
family (exponents 1/3, 2/3; the wall layer carrying the tail dies when
diffusion crosses it), with the one-point match a coincidence of the
parameter point. Three crossover extractors were needed before the sweep
could be trusted: the first measured the peak-depression dip, the second
fell to correlated-sample noise, and the third (cumulative excess mass,
parameter-free) reproduces the original 1.73 s measurement to 0.8% as its
consistency anchor. Both failures are recorded in the tool.

**Three solver legs, one referee** (`results/mc_channel*.json`, collated in
`results/mc_channel_benchmark.json`):

- **langevin** — exact kinematics at D = 0 (closed-form positions, binomial
  tolerances: RMSE at 1.06-1.27x its floor, pre-onset counts exactly zero);
  at physical D the tail departs in TWO regimes, in the OPPOSITE order to
  the prediction written beforehand: enhanced up to 1.6x at 5 t2 (the
  upstream reservoir, ~ dbar/c_x times the window population, is pumped in
  faster than the window drains), then terminated — exactly zero by 12 t2
  where the model predicts 1e-2. The wrong prediction stays in
  `runners/langevin.py` with its correction.
- **openfoam_particles** — the Hofmann replication rung: their published
  model has no diffusion, so the D = 0 run IS their model class
  (MPPIC-faithful replication is impossible in stock OF14, and the authors'
  DMPPIC source is deleted with no archive — both recorded). RMSE
  1.22-1.39x the binomial floor; the predicted one-sign interpolation bias
  is below resolution at N = 2e4 and recorded as unresolved. At physical D
  it CROSS-CONFIRMS the two-regime tail: enhancement 9.2x its noise bound,
  termination to exactly zero at 12 t2.
- **openlb** — the Eulerian scalar on the D3Q7 ADE lattice, prescribed
  Poiseuille, bounce-back walls. Stability PINS tau against 1/2
  (u_lat = (tau-1/2) c_s^2 Pe_cell; tau = 0.6 diverged, measured), which
  forces the corner where only the eigenvalue law D = c_s^2(tau-1/2)
  survives. OpenLB's converter realises the requested tau to 6 decimals.
  Peaks lag t2 by +4.0% at all three receivers; the far tail reads
  1.3-1.6x the flow-dominated reference — ABOVE the particle legs'
  physical enhancement, the excess being numerical dispersion at cell
  Peclet 33 (ringing minima to -9.5e-3 at the near receiver). Three
  instrumentation findings en route: `momenta::setDensity` never reaches
  bounce-back cells; stock `BounceBack` density reads a FIXED 1; the
  `BulkDensity` variant reads the Revert collision's period-2 cycle — so
  no density functor on bounce-back cells is mass accounting, and the CIR
  uses bulk-only sums with the parked-mass decline (-2.7%) named, bounded,
  and recorded.

The Tier-2 statement: what the two independent particle implementations
agree on is the physics; OpenLB's excess over that is its numerical
transport error at the parameter point stability forces on it. For
inter-symbol interference the flow-dominated model underestimates while
the tail is enhanced and overestimates after the termination — and its
Eulerian competitors carry the same limitation plus their scheme's own
dispersion.

## Conservation check on production code

`tools/identity_check.py` is a standalone, READ-ONLY flux-closure checker. It
is deliberately coupled to nothing — not to betaflow, not to AortaCFD, not to
any pipeline. It takes a case directory, reads the written `phi` field, and
reports whether the discrete continuity identity closes:

    sum over all boundary faces of phi = 0

It runs no solver and writes nothing into the case, so it is safe to point at
production output. Handles ascii and binary fields.

**Why this check and not another.** A mis-coupled outlet — a Windkessel with
the wrong sign, a flow split that does not sum to the inlet, an outlet left on
a default BC — produces a plausible velocity field, a plausible pressure
field, and a plausible WSS map. Nothing in those outputs reveals it. The flux
balance does, immediately. It is the same structural rule as everything else
here: an exact relation the discrete solution must satisfy, independent of the
physics being modelled. Multi-outlet coronary cases are where it matters most,
because the failure mode is likelier and harder to spot by eye.

**CORRECTION: the mass tier is weaker than first claimed.** In SIMPLE/PISO the
pressure equation is constructed so the corrected face flux satisfies
div(phi) = 0 in every cell to the p-solver tolerance; summing over cells,
interior faces cancel pairwise and what remains is the boundary sum. So
closure is a direct consequence of the pressure solve converging — OpenFOAM
already prints it every step as the global continuity error. It confirms the
pressure equation converged and is NOT independent of what the solver gates
on. A check implied by something already enforced is close to vacuous, which
is lesson 2 in another costume.

It also does NOT catch a mis-coupled Windkessel, contrary to the claim first
made for it: a wrong resistance changes the flow SPLIT between outlets while
total mass still balances exactly. It catches an outlet not participating at
all — a typo'd BC type, a truncated write — not one with wrong parameters. On
multi-outlet coronaries the named failure mode is precisely the one this tier
cannot see.

**Result on production output: clean.** Five cases across both pipelines, all
closing at the linear-solver tolerance rather than merely "small":

| case | throughput | relative imbalance |
|---|---|---|
| coronaryCFD SUK_BIF_0000_CPD10 | 6.46e-6 | 5.5e-8 |
| coronaryCFD SUK_BIF_0000_CPD14 | 6.49e-6 | 1.7e-7 |
| coronaryCFD SUK_BIF_0000_CPD20 | 6.50e-6 | 5.7e-8 |
| AortaCFD BPM120 validation_lesson01 | 1.66e-4 | 2.4e-7 |
| AortaCFD PAT_0000 | 1.64e-4 | 1.2e-8 |

A clean bill of health is worth having and worth stating plainly. This is the
original use #1 that justified building the framework, and the answer is that
the production code conserves mass.

**One reproducibility note, corrected.** Four of the nine cases examined have
no written time directories AT ALL — not specifically a missing `phi`, as
first reported. The five that do write output carry the complete set
(`phi`, `p`, `U`, `wallShearStress`). A quantity never written cannot be
audited, so keeping them is what makes any of this checkable after the fact.

### Momentum tier

**What it tests, stated precisely.** Summed over cells with the solver's own
discrete operators, the momentum equation telescopes to its boundary terms —
so a momentum balance built that way is implied by the solve, exactly as the
mass balance is. This tier does NOT claim independence from the solve.

Its value is different: **WSS as published does not come from the solver's
momentum assembly.** The `wallShearStress` / `forces` functionObjects
reconstruct it from snGrad and the viscosity model, a separate code path.
Comparing that reconstruction against the balance the solver enforced tests
the POST-PROCESSING CHAIN — which is where the published number comes from,
and which nothing else checks.

**FIFTH INSTRUMENT FAILURE, and it inverted the conclusion.** The first
version of this table normalised the residual by the GROSS per-patch
magnitude, which includes sum|p|A. That shifts to |p0| sum(A) under
p -> p + p0, so it is pressure-REFERENCE-dependent even though the identity
itself is not — the NET is invariant only because the closed-surface integral
of n vanishes, and that argument does not extend to a gross. Measured
references: the coronary cases carry ~89 mmHg with a pressure RANGE of 1% of
it; BPM120 carries ~37 mmHg with a range comparable to its mean. So the
normalisation was comparing pressure references, not physics.

Removing the area-weighted mean boundary pressure leaves the NET identity
untouched (same closed-surface argument) and makes the scale comparable:

| case | mean p | reference-DEPENDENT worst | reference-FREE (x, y, z) | inflation |
|---|---|---|---|---|
| AortaCFD PAT_0000 | 10.12 | 4.6e-4 | 1.8e-3, 3.3e-3, **3.8e-2** | 83x |
| AortaCFD BPM120 lesson01 | 4.69 | 4.8e-3 | 1.2e-2, 8.9e-3, 6.3e-3 | 2x |
| coronaryCFD SUK_BIF_CPD10 | 11.18 | 1.9e-5 | 3.6e-4, 8.5e-4, 2.5e-3 | 131x |
| coronaryCFD SUK_BIF_CPD14 | 11.18 | 1.2e-5 | 4.4e-4, 1.6e-3, 1.5e-3 | 130x |
| coronaryCFD SUK_BIF_CPD20 | 11.18 | 1.1e-5 | 2.0e-4, 1.4e-3, 9.6e-4 | 128x |

**SIXTH INSTRUMENT FAILURE — trap 4, which was specified and never
implemented.** The two AortaCFD cases run `ddtSchemes: Euler`; all three
coronary cases run `steadyState`. The checker applied the STEADY momentum
identity to transient cases, silently omitting d/dt of the volume-integrated
momentum. Computing that term (cell volumes from the mesh, finite-differenced
across the last two writes) accounts for PAT_0000's dominant z residual to
**4%** (-4.196e-6 measured against -4.386e-6 omitted), and cutting it drops
PAT_0000 from 3.8e-2 to **3.4e-3** — an order of magnitude, into the coronary
range. It does NOT touch BPM120, whose d/dt is ~1e-8, a thousand times too
small to explain its residual.

So the ordering inverts TWICE, and lands where it started:

| case | scheme | pressure-dependent | reference-free | + transient term |
|---|---|---|---|---|
| BPM120 lesson01 | transient | 4.8e-3 | 1.2e-2 | **1.2e-2** |
| PAT_0000 | transient | 4.6e-4 | 3.8e-2 | **3.4e-3** |
| coronary CPD10/14/20 | steady | ~1e-5 | 1.4-2.5e-3 | 1.4-2.5e-3 |

**BPM120 is the genuine outlier after all** — but the first analysis was right
by accident, two instrument errors having pointed opposite ways and partially
cancelled. (CORRECTED: this sentence said "the same way", which cannot produce
cancellation; PAT_0000's datum-free renormalisation raised its residual and
the omitted transient term's restoration lowered it.) And BPM120 is the harder finding: its flow has reached a steady
state (d/dt ~ 1e-8) despite a transient scheme, so the steady identity SHOULD
apply to it, and it does not close. That is the one result here still
unexplained, and the one worth investigating before publication.

The now-superseded reference-free-only conclusion read:
the coronaries were flattered by ~130x purely by their pressure offset. And
PAT_0000's z-component is 10-20x its own x and y, so the per-component flag
was pointing at the real thing all along rather than incidentally. Closed
surface stays at 1e-16 to 3e-16 throughout, and the raw residual is unchanged
by the renormalisation — only what it is divided by.

Closure is at the momentum-residual level, decades looser than the mass
tier's 1e-8, as predicted: the momentum residual is the binding constraint,
not the pressure solve.

**Wall traction is NOT wall shear stress** — but the |p|/|visc| ratio of
400-700x first quoted here is NOT a quotable physical number, for the same
reason: it is dominated by the pressure reference. The physical statement
that survives is qualitative and still worth making: the traction carries a
-p n part that does not vanish on a curved wall, so integrating
`wallShearStress` alone and calling it the wall force is wrong. Quantifying
"by how much" requires a stated pressure reference, and every case here
carries a different one.

**Two traps that bit during construction, both worth recording.** First, `p`
is `zeroGradient` on walls and inlets, so its boundary values are never
written; omitting those patches left the traction integral
reference-dependent and the residual at 0.08-0.21 — an apparent physics error
that was pure instrument. Reconstructing the face value from the owner cell
(exact for zeroGradient) brought closure to the levels above. Second, the
normalising scale must be the GROSS per-patch magnitude, not the net: a
Couette channel's two walls cancel by construction, and dividing by that net
reports a relative residual of 1.0 for a perfect balance.

### The instrument has its own null test

`tests/test_identity_checker.py`, in the analytic reference CI tier — pure file parsing
against a committed 76 kB fixture, no solver, milliseconds. It checks the
closed surface (round-off), the volume from the boundary alone
(V = 1/3 closed-integral x.n dS, exact), mass closure, and the momentum
identity on a case whose answer is known exactly (tau_w = G h), requiring
**per-component residuals below 1e-12**. Measured on the three betaflow null
cases: **8.8e-17 (channel), 8.3e-17 (Couette), 1.25e-12 (pipe)** — the pipe value
matching the N=80 level's own identity residual (1.24997e-12 in
results/pipe.json) to all printed digits, so that is the case's convergence
rather than the instrument. (CORRECTED: this line previously paired the
1.25e-12 with the N=40 value 2.5e-12 and called it a match — a factor-of-2
gap glossed as agreement.)

This exists because the mass parser's first run reported a spurious 33%
imbalance, caught only because 33% is implausible; the same bug at 1e-4 would
have read as mild under-convergence and been believed. An instrument needs
calibrating against a known answer before it is pointed at unknown data —
the analytic-analytic reference argument, one level up. It earned its keep twice more
during this work: a display artefact printed BPM120's residual as exactly
[0, 0, 0], which investigation turned into the worst result of the five.

**And here is the limit of a null test, stated because Stage A below is
exactly it.** A null test certifies the instrument only in the CONFIGURATION
it was run in. Every null case above drives flow with cyclic streamwise
patches and a body force; in that configuration the inlet and outlet
contributions to the momentum balance cancel identically, so the
pressure-force and momentum-flux terms were never exercised against a known
answer — and those terms carry the entire balance in a patient case with a
prescribed inlet and Windkessel outlets.

### Stage A: the null test in the configuration production actually uses

`betaflow/cases/pipe_poiseuille_io.yaml` closes that gap while keeping an
exact answer. A straight pipe on a circumscribed wedge; the exact developed
Poiseuille profile written at the true face centroids after `blockMesh`; fixed
outlet pressure; no body force. The flow is developed throughout so momentum
flux cancels between the ends, and the balance reduces to
(p_in − p_out)·A = tau_w·2·pi·a·L.

**It does not close at the level cyclic cases reach.**

| nx × nr | inlet p-force | wall viscous | flux | residual | relative |
|---|---|---|---|---|---|
| 8 × 40 | 0.0278271 | −0.0278882 | 4.70e-07 | −6.161e-05 | **2.214e-03** |
| 16 × 80 | 0.0278761 | −0.0279257 | −1.51e-06 | −4.808e-05 | **1.725e-03** |
| 32 × 160 | 0.0279074 | −0.0279378 | −5.95e-07 | −2.986e-05 | **1.070e-03** |

The velocity converges at **2.04 then 3.02** against the exact profile on the
same three runs.

**"Floor" was not merely unsupported — it was the wrong model.** The apparent
orders are 0.36 then 0.69, *increasing*, which is a floor's opposite
signature; fitting r = C + A·h^p to the three levels returns
C = **+3.654e-03**, larger than all three measured residuals, so the model
contradicts the monotone decrease that motivated it. The structural reason
came later: a residual formed as the difference of two contributions
converging at DIFFERENT rates has no single apparent order, and r = C + A·h^p
has no term that can represent one rate cancelling against another.

**All three explanations tested, with no new solve.**
`tools/wall_traction_compare.py` (read-only), `results/stage_a_discriminators.json`.

| hypothesis | test | outcome |
|---|---|---|
| (a) the checker's surface quadrature | wedge areas and volume against their closed forms | **REFUTED** — 9e-16 at every level, including the END faces no cyclic case can exercise |
| (b) `wallShearStress` FO ≠ the solver's momentum assembly | wall force from the FO, differenced against nu·snGrad(U) on the solver's own cell values and cell centres | **REFUTED** — 7.3e-07, 8.2e-08, 3.0e-09 (orders 3.16, 4.80) |
| (c) incomplete development | inlet-to-outlet change in the radial profile | 8.7e-03, 3.2e-03, 9.2e-04 (orders 1.46, 1.78) — second order, not the residual's rate |

(b) is the one that mattered. Had the post-processing chain disagreed with the
solver's momentum assembly at 1e-3, every published wall-shear number from
either production pipeline would inherit that discrepancy. It does not.

**What it actually is: pressure-velocity decoupling at the inlet.** The
discrete pressure carries a period-2 oscillation in the streamwise index —
measured both along a single radial line and on the cross-sectional mean, so
it is not an averaging artefact. Amplitude near the inlet falls at FIRST order
(0.89, 0.96) while it damps downstream (2.33, 3.88), and the wall viscous
force converges at second (1.66, 1.74). A momentum balance over the downstream
half, whose upstream face is an interior plane using the solver's own
interpolation and its own face fluxes, closes **16.3x better than the
full-domain balance** at the finest level (1.07e-3 vs 6.56e-5;
`ratio_at_finest` in the record). CORRECTED: this line said 8x, which is
the interior balance's own coarsest-to-finest improvement — a different
comparison — and the record carried the same contradiction internally. Its own apparent orders are 0.93 and 2.07 — not constant over three
levels, so **8x is the quotable number and the rate is not**. Rhie-Chow
momentum interpolation suppresses the mode in the interior and is weakest at a
boundary where pressure is specified and velocity is not, which is where the
oscillation is largest and why excluding that one patch accounts for most of
the residual.

**Consequence.** The ~1e-3 is NOT an instrument artefact, so the production
momentum residuals above are not explained away by it. Instrument failure #7
is resolved: the checker was not at fault, it was reporting a real defect.
And the pairing is the strongest statement this repo makes — one solution, one
run, an exact analytic reference, a textbook second-order convergence study
that passes cleanly, and a real defect that only the conservation identity
sees. The two earlier instances both have an escape route (the kernel swap
compares two *different* governing solutions; womersley_carreau has no exact
analytic reference). This one has none.

**NOT reproducible from the repo as it stands, and this is the first thing to
fix here.** The three runs came from an ad-hoc script: the committed runner
fixes `_N_STREAMWISE = 4` while the meshes on disk are 8/16/32 streamwise.
There is no `tests/test_pipe_io.py`. The case YAML and the results are
committed; the path that produced them is not.

## Adding a case

1. Write the analytic reference in `betaflow/analytic/` — a pure function returning
   non-dimensional profile values, with the Reynolds-number definition (length
   AND velocity scale) stated in the docstring.
2. Add `betaflow/cases/<name>.yaml`: geometry, `nondim`, dotted `analytic reference` path,
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
