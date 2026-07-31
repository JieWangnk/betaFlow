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
- **Solution verification** — no exact solution (patient geometry, anything
  genuinely multi-dimensional). There the discretisation error can only be
  *estimated*, and GCI enters. GCI is never used where an oracle exists.

**Correction (superseded claim).** An earlier version of this file listed
Carreau rheology as solution-verification-only, "no exact solution". That was
wrong. In steady fully-developed 1-D channel flow the momentum equation
integrates once to

    tau(y) = G y        EXACTLY, for ANY rheology

— the same force balance that makes tau_w rheology-independent. So a
machine-precision oracle exists for *every* generalised Newtonian model: solve
the scalar monotone equation nu(gammadot)·gammadot = G y pointwise, then
integrate gammadot to get u. No ODE solve, no shooting, no GCI. `carreau_steady`
is therefore code verification with an order-of-accuracy test, and its oracle
self-verifies against the Newtonian and power-law limits to 1.8e-16 and
7.2e-16 before being used as ground truth. The same construction would cover
Herschel-Bulkley, Cross, and power-law without new machinery.

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

    python3 -m pytest -m oracle -v   # oracles only, NO solver needed (~0.4 s)
    python3 -m pytest tests/ -v      # default: skips @slow studies (~9 min)
    python3 -m pytest -m "" -v       # everything (21 tests, ~32 min)
    python3 -m pytest -m slow -v     # the slow studies alone

Tiering is by pytest marker (`addopts = -m 'not slow'` in pyproject.toml) and
wired into `.github/workflows/ci.yml`: the oracle tier gates every push with no
solver install, the default tier runs on push, the full tier runs nightly. The
casson two-axis grid and womersley_carreau are marked slow. `langevin_free`
needs no solver at all and runs in ~5 s.

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
| womersley_carreau | identity_max_rel | tau_w(t) = h(G - d<u>/dt) — NO profile oracle | N=336, nt=1024 | 2.7e-7 | 1e-6 |
| carreau_steady | L2_velocity | Carreau-Yasuda channel (rootfind + quadrature) | N=160, Cu=10 | 6.1e-5 | 1e-2 |
| carreau_steady | wss_relative | tau_w = G h (rheology-independent) | N=160, Cu=10 | 0.0 | 1e-6 |

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

## carreau_steady: the contrast case

Carreau has no regularisation parameter — nu is smooth and bounded between
nu_inf and nu_0 at every strain rate — so no cap, no plug, no second axis.
Committed predictions and outcomes (`results/carreau_steady.json`):

| prediction | outcome |
|---|---|
| tau_w = G_disc·h to round-off | **confirmed** — identity exactly **0.0** on most runs, ≤2e-12 on all |
| p ≈ 2 and NO floor | **confirmed** — p = 1.979, 1.990; error 9.50e-4 → 2.41e-4 → 6.07e-5, still falling |
| limits recovered | **confirmed** — oracle self-verifies to 1.8e-16 (Newtonian) and 7.2e-16 (power law) |
| no stiffening | **partly wrong** — geometric, but cost grows as ~Cu^0.6 (below) |
| flatness monotone in Cu | **wrong, and the oracle agrees** (below) |
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

**Two predictions failed for one shared reason, and the exact oracle confirms
it is physics, not a solver artefact.** Centreline flatness u_centre/u_mean
over the sweep is 1.4978 → 1.4635 → 1.3450 → **1.3875**: non-monotone, turning
back up at Cu = 100. The oracle gives 1.4996 / 1.4655 / 1.3470 / 1.3898 —
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

## womersley_carreau: verification without an oracle

The first case with NO exact solution — the point of it. The unsteady term
breaks the force balance every previous oracle rested on, and nu(gammadot)
kills superposition, so the profile has no closed form. Patient geometry has
no oracle either; this is the rehearsal for that.

**Verified EXACTLY** (measured, not estimated):

| quantity | result |
|---|---|
| momentum identity tau_w(t) = h(G(t) - d<u>/dt), per timestep | 2.72e-07 |
| half-wave symmetry u(y,t+T/2) = -u(y,t) | 12.1x the periodicity residual (see below) |
| Cu -> 0 vs the exact Womersley cosh oracle | L2 amp 1.02e-03, phase 4.17e-04 rad |
| alpha -> 0 vs the exact steady Carreau oracle | L2 1.99e-03 |

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
profile SHAPE still matches the steady oracle to 2e-3.

## Pipe geometry: what the harness assumed

Three cases were ported to a circular pipe on an axisymmetric wedge —
`pipe_poiseuille_steady`, `pipe_casson_steady`, `pipe_womersley_pulsatile` —
to answer one question: are the abstractions geometry-agnostic, or quietly
channel-specific? **Quietly channel-specific, in seven places.** All seven are
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
metric needed **no change at all**, and the oracles transferred as pure factor
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
| 2 | 1.69e-1 | 3.89e-1 | 3.40e-1 |
| 5 | 6.25e-2 | 5.22e-2 | 4.64e-1 |
| **10** | **9.14e-3** | **1.46e-2** | **4.82e-1** |
| 20 | 2.59e-3 | 4.56e-3 | 4.91e-1 |

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

CITATIONS. Every pipe equation traces to a source recorded in both the oracle
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

**Three distinct sigma laws now exist in the harness and must not be
interchanged**: 0.8165/sqrt(N) for a single-time 3-D MSD, 0.7071/sqrt(N) for a
through-origin slope fit, and sqrt(2/N) for a variance estimator such as an
axial dispersion coefficient. Baking any one of them in as "the Monte Carlo
error law" is the statistical analogue of reusing a channel oracle on a pipe.

Fluctuation-dissipation is asserted directly: the friction coefficient in the
noise amplitude and in Stokes drag must be one and the same zeta = 6 pi mu a
(round-trip error 0.0). Taking them from independent constants is the classic
implementation bug, and it silently changes the MSD slope.

Overdamped (inertialess) Langevin is justified by St = 5.0e-9 at these
scales, reported in meta rather than assumed.

## taylor_aris: the last exact oracle

Dispersion of a nanoparticle in laminar pipe flow, on `runners/langevin.py`
with an analytic Poiseuille field — no CFD solver. After this there are none:
margination, capture efficiency and deposition have no closed forms, so this
is the final rung where a result can be checked against truth rather than
estimated.

The oracle self-verifies before use, at machine precision: Decuzzi Eq. 15
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
original use #1 that justified building the harness, and the answer is that
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

The ordering INVERTS. **PAT_0000 is the worst case, by 3.3x**, not BPM120;
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

`tests/test_identity_checker.py`, in the oracle CI tier — pure file parsing
against a committed 76 kB fixture, no solver, milliseconds. It checks the
closed surface (round-off), the volume from the boundary alone
(V = 1/3 closed-integral x.n dS, exact), mass closure, and the momentum
identity on a case whose answer is known exactly (tau_w = G h), requiring
**per-component residuals below 1e-12**. Measured on the three betaflow null
cases: **8.8e-17 (channel), 8.3e-17 (Couette), 1.25e-12 (pipe)** — the pipe
matching its own case identity of 2.5e-12, so that is the case's convergence
rather than the instrument.

This exists because the mass parser's first run reported a spurious 33%
imbalance, caught only because 33% is implausible; the same bug at 1e-4 would
have read as mild under-convergence and been believed. An instrument needs
calibrating against a known answer before it is pointed at unknown data —
the analytic-oracle argument, one level up. It earned its keep twice more
during this work: a display artefact printed BPM120's residual as exactly
[0, 0, 0], which investigation turned into the worst result of the five.

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
