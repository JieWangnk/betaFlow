# betaFlow results report

**Date:** 2026-08-21 · **State:** all results committed through `43a00ae` · **Suite:** 55 tests (27 analytic-tier, ~3 s; 45 default; 10 slow) ·
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
(`betaflow/analytic/channel_impulse.py`, 18 self-checks). Its
log-divergent tail is usable for at most 8.5 / 6.2 / 5.0 peak-times at
the paper's three receivers; the crossover clock, measured by a
pre-registered Peclet sweep (5 Pe values × 3 seeds × 3 receivers), is
**t_cross = K·τ_r^0.31·dbar^0.73** — the layer-escape scaling (predicted
exponents 1/3, 2/3; both matched within 0.07, rms log-residual 0.018, at
seed-scatter level). A first attribution to the relaxation eigentime
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
   wrong in shape as well as size; measured 8.5/6.2/5.0).

---

## What is next

1. **The coupled channel model** — OpenLB solving its own flow while
   advecting the scalar; today's wall measurement says use Bouzidi for the
   fluid lattice or budget the dx^1.4 radius bias. This is paper 1's
   production configuration.
2. **The layer-escape O(1) constant** — the measured prefactor is 2.78 ±
   0.17 times the crude balance; a proper derivation is open theory work.
3. **Comms-rate metrics** (inter-symbol interference in symbol terms,
   achievable rate) on top of the CIR; then bifurcating geometry.
4. **Paper loose ends needing the author:** §1 exemplar citations, the
   Rhie & Chow reference, the author block; BPM120's outlier.
