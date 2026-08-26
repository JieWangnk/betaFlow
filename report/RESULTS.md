# betaFlow results report

**Date:** 2026-08-26 · **State:** all results committed through `89d8b71` · **Suite:** 58 tests (27 analytic-tier, ~5 s; 46 default; 12 slow) ·
**CI:** analytic and solver jobs green on the pushed state.

Every number below traces to a committed record in `results/`, named in
place. To regenerate any of them: the record's test or tool re-runs it
(`docs/DESIGN.md` maps case → test → record). Corrections are part of the
results and get their own section — a withdrawn claim stays in the record
next to its replacement.

**What betaFlow is.** A validation framework for flow solvers: a library of
flow problems whose answers are known exactly, run through six solver
adapters (OpenFOAM fluid, OpenFOAM particles, a Python Langevin walker, a
moment integrator, a minimal lattice-Boltzmann code, and OpenLB), with
error measures and tolerances declared in case files and every tolerance
carrying a stated origin. Work is organised as a three-tier ladder:
re-examine published claims with exact mathematics (Tier 0), replicate one
solver at a time against the exact answer (Tier 1), benchmark solver
against solver with the exact answer as referee (Tier 2).

---

## Tier 0 — published claims re-examined, solver-free

**The kernel blind spot (paper §3).** The pulsatile-flow solutions for a
pipe and a flat channel are easy to swap. At Womersley number 10 the
swapped kernel misfits the velocity profile by 0.91% in amplitude and
1.46e-2 rad in phase — inside ordinary validation tolerances — while
misfitting wall shear stress by 48%. A solver can pass profile validation
while solving the wrong geometry's equations, and the discrimination ratio
grows with Womersley number. `results/kernel_discrimination_scaling.json`,
`pipe.json` (the study is re-run inside `tests/test_pipe.py` so it can
never drift from the claim).

**The molecular-communications model's validity clock.** The
flow-dominated channel-impulse-response model (Hofmann et al. 2024,
Eq. 13) was re-derived independently and confirmed
(`betaflow/analytic/channel_impulse.py`, 23 self-checks). Its
log-divergent tail is usable for at most 8.3 / 6.1 / 4.9 peak-times at
the paper's three receivers; the crossover clock, measured by a
pre-registered Peclet sweep (7 Pe values 50–3200 × 3 seeds × 3
receivers, 18 seed-averaged points spanning nearly two decades of τ_r),
is **t_cross = K·τ_r^0.31·dbar^0.71** — the layer-escape scaling
(predicted exponents 1/3, 2/3; both matched within 0.05, rms
log-residual 0.026, at seed-scatter level). A first attribution to the relaxation eigentime
τ_r/β₁² rested on a one-point match (0.95) and was refuted by the sweep.
`results/eigentime_pe_sweep.json`, `hofmann_validity_audit.json`.

**Why grid methods struggle at these parameters.** At the anchor paper's
Pe = 200, first-order upwind finite volume carries artificial diffusion
from 10× the physical value (5 cells per radius) to 0.5× (100 cells per
radius): the scheme's own artefact rivals the physics at any affordable
mesh — the quantitative reason this field uses particle methods or LBM.
The same computation at haemodynamic parameters gives an artefact fraction
of 0.99999. `results/hofmann_validity_audit.json`;
`betaflow/analytic/numerical_diffusion.py` (16 self-checks).

**Lattice-Boltzmann transport laws, derived then measured.** The LBM
reference (`analytic/lattice_boltzmann.py`, 95 self-checks) derives, among
others: the diffusivity law D = c_s²(τ−½) whose naive form is wrong by
51× at τ = 0.51; the first-order-equilibrium depletion
D_eff = (c_s² − u²)(τ−½); the wall-slip law with its zero at Λ = 3/16 —
and documents a sign error in the published version (arXiv:1603.09577,
Eq. 73). The mini-LBM runner then measured the slip zero-crossing at
τ = 0.9330127019, against the exact ½+√3/4 = 0.9330127019 — agreement to
ten digits. `results/lbm_scalar.json`.

**The off-axis Ma² tensor (new).** The depletion law's diagonal-flow
conjecture is resolved: the BGK ADE scheme's exact k→0 diffusion tensor
is D_ab = (τ−½)(Π^eq_ab/C − u_a u_b) for every tabulated velocity set,
both equilibrium orders, any flow direction — verified against the exact
amplification matrix at 50-digit precision (80 cases, worst 5.6e-17; the
LBM reference now carries 119 self-checks) and confirmed on an actual
D2Q5 lattice to 9.4e-8 of D_xx. Three consequences with teeth: the
first-order depletion is rank-one along the flow (transverse diffusion
exactly undepleted); oblique advection generates a *negative* cross
diffusion −(τ−½)u_x u_y; and on the reduced sets — OpenLB's ADE
lattices — that cross term survives the second-order equilibrium fix at
full size, because no D2Q5/D3Q7 velocity carries the fourth moment the
fix routes through. This was the blocking prerequisite for the
bifurcation rung, whose daughter branches run oblique to the lattice;
the budget line is in `docs/bifurcation-preregistration.md`.
`results/lbm_offaxis_tensor.json`.

**OpenLB first contact.** OpenLB 1.9's shipped advection-diffusion
benchmark realises D_eff = 0.908 of its requested 1.5 and u_eff = 9.09 of
its requested 10; the depletion law predicted the exact-eigenvalue values
(0.9031, 9.102) before measurement — agreement 0.5% and 0.1%.
`results/openlb_first_contact.json`.

---

## Tier 1 — one solver at a time against the exact answer

**OpenFOAM, fluid.** The Couette null test sits at round-off
(L2 ≤ 1.6e-14, wall shear ≤ 5.1e-13 across three mesh levels) — the
instrument is clean before it measures anything. Channel Poiseuille:
profile error 3.3e-4 at N = 80, wall shear 3.1e-4 (a genuine accuracy
check under velocity-targeted driving). Pipe Poiseuille: observed
convergence orders 1.97 and 1.98 (band 1.8–2.2), and the force-balance
identity τ_w = G·a/2 at 1.2e-12 to 5.0e-12 — flat across refinement, the
identity-pinned signature; this check historically caught a
refinement-independent wedge-faceting bias (9.5e-4) and a 100% lever-arm
error. Casson and Carreau rheology and pulsatile Womersley flow pass with
their identities; the regularisation study (paper §5) showed residuals
stall while the conservation identity still measures distance.
`results/couette_steady.json`, `poiseuille_steady.json`, `pipe.json`,
`casson_steady.json`, `carreau_steady.json`, `womersley_pulsatile.json`.

**OpenFOAM, particles.** Free Brownian motion: MSD slope error at 1.1σ of
its own statistical law (N = 1e5). Taylor–Aris dispersion in a frozen
Poiseuille pipe: D_eff error 3.3e-3 (0.27σ), radial-distribution gate at
0.51 of its KS floor — the gate that caught two wall-scheme defects during
development. `results/langevin_free_openfoam.json`,
`taylor_aris_openfoam.json`.

**Langevin (pure Python).** The reference implementation for particle
transport: Taylor–Aris D_eff/D = 9.9997 against the exact 10.0, and the
comms case's exact-kinematics rung within pure counting noise (profile
RMSE at 1.06–1.27× the binomial floor, pre-onset counts exactly zero).
`results/taylor_aris.json`, `mc_channel.json`.

**OpenLB, scalar (D3Q7).** The comms case as an Eulerian scalar with
prescribed flow: stability pins τ to 0.5048 (any comfortable τ diverges —
measured), peaks lag by exactly +4.0% at all three receivers, mass
parked in bounce-back walls bounded at −2.7% with the mechanism named,
and three instrumentation traps recorded (no density read-out on
bounce-back cells is mass accounting). `results/mc_channel_openlb.json`.

**The oblique-wall instability, and the G4 machinery gate (new).** The
bifurcation programme's first gate (the bent-pipe control,
`docs/bifurcation-preregistration.md`) caught a solver limitation before
any junction run: stock BGK ADE at the stability-pinned τ is **unstable
whenever advection runs oblique to the lattice** — a straight pipe tilted
30° diverges with no bend present (bulk mass −1.9e51). The
discriminating ladder (mitre, smooth arc, oblique straight; doubled
damping margin; second-order equilibrium; TRT with even-sector damping;
TRT at magic Λ) excluded, in turn, the elbow construction, the damping
margin, the equilibrium order, and the even sector; the uniform scheme is
plane-wave stable at every operating point (full 3-D wavevector scan,
re-derived in-test), which localises the mechanism to the wall: oblique
flow drives advective flux through the staircase's bounce-back faces —
exactly zero in the axis-aligned case, which is why the straight
programme never saw it. TRT at Ginzburg's Λ = 1/4 reduces the gain by
orders yet the loop persists at production resolution; stock
OpenLB 1.9's only interpolated ADE wall is absorbing; and plain Bouzidi
reflection is refuted by direct probe (stable axis-aligned but retains
0.25 of the scalar — the interpolation is not conservative, and a scalar
has no pressure field to self-correct). So **stable oblique scalar
transport awaits a mass-conserving interpolated no-flux ADE wall — the
named next rung; no bend number is quoted before it lands.** What DID
pass: the machinery gate — the junction code path (composite arc
geometry, region-wise velocity, path windows, capped ends) reproduces
the straight record to 0.007% on peaks once a release-slug edge-rounding
flakiness the gate itself surfaced was fixed (the straight app's slug
boundary sits exactly on cell centres; 3 slices there against 2 in the
new app was a 50% release-width difference wearing a +4% peak
discrepancy). The TRT collision change is measured, mechanism named
(finite-k diffusivity separation, ratio 0.83 at k = 1): peaks
+0.4/+6.2/+5.0%, tails within the envelope.
`results/bifurcation_g4_control.json`.

**OpenLB, momentum (D3Q19) — the wall-position measurement.** The same
pipe case that examined OpenFOAM, run through OpenLB's fluid solver with
the wall treatment as the variable. Bounce-back: the effective radius sits
**inside** the geometric one and the offset decays as a_eff − a ≈ −dx^1.4
(−0.41 / −0.31 / −0.24 dx at N = 21/41/81), error order 1.4–1.3 — the
staircase signature; τ moves the shift only 3% over the stability-allowed
range, so resolution dominates τ on a staircase cylinder. The Bouzidi
control on identical runs: shift ∝ dx² (−0.050 / −0.025 / −0.011 dx),
order 2.1 — confirming the instrument and isolating the wall (both walls
share compressibility and the convergence budget; a doubled-time pair
moved the shift by under 5e-5 dx). This measurement closes the oldest declared
UNRESOLVED item in the LBM reference for this configuration. Design
consequence: prefer Bouzidi walls for the coupled model's fluid lattice,
or budget a dx^1.4 radius bias. `results/openlb_wall_position.json`,
`pipe_openlb.json`.

---

## Tier 2 — the benchmark: three solvers, one referee

The comms case (`mc_channel`: Hofmann's Table-1 pipe at Pe = 200) runs
through three independent implementations against one exact solution,
collated in `results/mc_channel_benchmark.json`:

| Leg | Replication rung | Departure at physical D | Cost |
|---|---|---|---|
| Langevin | RMSE 1.06–1.27× binomial floor; pre-onset exactly 0 | peaks 0.93–0.94 of model; tail enhanced then terminated | ~7 s |
| OpenFOAM particles | RMSE 1.22–1.39× floor; pre-onset exactly 0 | same two-act tail: enhancement 9.2× its noise bound, zero at 12 t₂ | ~4 min |
| OpenLB (Eulerian) | — (slug release) | +4.0% peak lag; tail 1.3–1.6× model — above the particle legs, the excess being numerical dispersion | ~2 min |

**The physics finding the benchmark established.** At physical diffusion
the flow-dominated model fails in two acts, in the opposite order to the
prediction written beforehand: the tail is first **enhanced** (1.67×
at 5 peak-times, record field ratio_at_5_t2 — the upstream reservoir of
slower particles, dbar/c_x = 7.5 times the window population, is pumped in faster than the window drains), then
**terminates completely** (measured exactly zero by 12 peak-times, where
the model still predicts 1e-2). Two fully independent particle
implementations agree on both acts, so the departure is physics; OpenLB
reproduces the direction with a larger magnitude, and the excess is its
scheme's dispersion at the stability-forced parameter point. For
inter-symbol interference the analytic model first under-estimates, then
over-estimates. `results/mc_channel_departure.json`,
`mc_channel_openfoam.json`, `mc_channel_openlb.json`.

**The coupled channel model (new).** The same case with OpenLB solving
the flow it advects on — the case YAML now states the fluid physics:
water, Re = 0.60, Sc = 667. The runner stages the lattices (converge the
D3Q19 fluid with Bouzidi walls per the wall measurement, freeze the
solved profile, run the scalar on it); staging is exact for steady flow
and avoids the Sc = 667 shared-step trap (tau_fluid would be ~2.9). The
fluid stage passes its own exam (L2 3.5e-3, shift −0.021 dx), and the
measured cost of the solved flow on the CIR is small: peaks 0.1–0.3%
below the prescribed leg, tail ratios within 0.04.
`results/mc_channel_openlb_coupled.json`.

**Comms-rate consequences (new).** The two-act tail changes the
communications numbers computed from the CIR. Worst-case on-off-keying
interference is the sum of every earlier symbol's tail at the detection
instant; under the analytic model every term sits on the c_x/(2Vt) tail,
so the sum grows as (c_x/2VT_s)·ln K without bound — the model cannot
define a worst-case interference or a channel memory at any signalling
rate (proved in closed form in `channel_impulse.py`, including the O(1/K)
truncation bound). The measured termination makes both finite: memory is
8 symbols at T_s = t₂ for the middle receiver. Truncating the model at
the same data window for a like-for-like sweep, the model *overstates*
interference at the near and middle receivers at every sampled interval
(its phantom tail keeps contributing particles that have physically
left), and at a 10% interference bound it certifies no rate at all for
the near and far receivers where the measured channel supports 1.93 and
0.37 bit/s, and understates the middle receiver by 1.6× (0.39 vs 0.61
bit/s). The pre-registered direction picture was too simple and is kept
per policy (see the corrections trail).
`results/comms_rate_metrics.json`.

**The Hofmann replication claim.** Their published model contains no
diffusion, so the diffusion-free rung IS the replication of their model
class, and it matches within counting noise. Exact MPPIC fidelity is
impossible in stock OpenFOAM 14 (the Brownian force only registers for
particle families that solver cannot build) and the authors' DMPPIC source
is deleted from GitHub with no archive — both recorded as citable negative
findings. `results/mc_channel_openfoam.json`.

---

## Production audit (read-only, real patient cases)

The identity instruments pointed at production output: three coronary
cases hold their momentum identities at 2.0e-4 – 2.5e-3; the corrected
transient form brought PAT_0000's worst residual from 3.8e-2 to 3.4e-3;
BPM120 remains the unexplained outlier at ~1.2e-2.
`results/production_identity_audit_transient_corrected.json`.

---

## The corrections trail (results in their own right)

The framework's standing rule is that corrections stay in the record with
their reasons, because how errors present is data. The trail so far:

1. **Eigentime crossover attribution** — withdrawn. A one-point match to
   a named constant (0.95 of τ_r/β₁²) dissolved under the pre-registered
   Pe sweep; the layer-escape scaling replaced it. Two crossover
   extractors failed before the sweep could be trusted (one measured the
   peak-depression dip; one fell to correlated-sample noise); the final
   integral extractor reproduces the original measurement to 0.8% as its
   consistency anchor. `results/eigentime_pe_sweep.json`.
2. **Depleted-tail prediction** — wrong in direction; the measured
   structure is enhancement first, termination after. Kept in
   `runners/langevin.py` beside its correction.
3. **Bounce-back wall-sign prediction** — wrong twice (outside vs inside;
   fixed fraction vs decaying dx^1.4). Kept in `openlb_cases/pipeFlow3d`
   and the test docstring. The Bouzidi control confirmed as predicted.
4. **Earlier corrections on record:** the 8× → 16.3× discriminator, the
   withdrawn error-floor claim, the mis-attributed dispersion constant,
   the 749/750 off-by-one that wore a plausible physics story, and the
   audit's eigentime-based validity extents (27.2/6.8/3.4 peak-times,
   wrong in shape as well as size; measured 8.3/6.1/4.9), and the
   pre-registered slot-position picture for interference direction (too
   simple; the whole-tail mass balance decides it, kept in
   `results/comms_rate_metrics.json`).

---

## What is next

1. **The layer-escape O(1) constant** — the measured prefactor is 2.73 ±
   0.16 times the crude balance; a proper derivation is open theory work.
2. **Bifurcating geometry** — pre-registered 2026-08-26; gate G4's
   machinery control PASSED (0.007% peaks) and surfaced the oblique-wall
   instability (see Tier 1); the naive interpolated candidate is refuted
   by probe. Next concrete step: design a mass-conserving interpolated
   no-flux ADE wall scheme, verified on the same G4 control before any
   bend or junction measurement.
3. **Paper loose ends needing the author:** §1 exemplar citations, the
   Rhie & Chow reference, the author block; BPM120's outlier.
