# betaFlow design: the verification hierarchy

betaFlow measures whether CFD solvers get known answers right, before anyone
trusts them on unknown ones. Every case pairs a solver run against an exact
solution, every tolerance has a stated origin (a derivation or a measured
statistical law), and every quoted number traces to a committed,
provenance-stamped record. The project exists because the errors that matter
are the ones that leave the solution looking correct — and those are found by
structure, not by care.

## Terminology

The modules in `betaflow/analytic/` are **analytic references**: exact
solutions written as pure functions, each carrying its citation and a
`verify_limits()` self-check that re-derives its constants by an independent
route before the module may serve as truth. The solver-free test tier is the
**analytic tier** (`pytest -m analytic`).

Migration note (August 2026): these modules previously carried a different
name throughout the repository, retired by a standing user language rule.
The rename commit changed the pytest marker (now `-m analytic`), the
case-YAML key (now `reference:`), the test filename (now
`tests/test_analytic.py`), and the result-record keys, with no behaviour
change. The old spellings are visible in that commit's diff and in every
earlier revision; they are not repeated here.

## The hierarchy: replicate, re-examine, benchmark

The project's shape is a three-tier ladder. Nothing on a higher tier is
trusted until the tiers below it are green.

**Tier 0 — RE-EXAMINE (solver-free).** Analytic references re-derive
published claims before any solver runs. Examples already on record: the
molecular-communications channel impulse response (Hofmann et al. 2024,
Eq. 13) re-derived independently and confirmed
(`betaflow/analytic/channel_impulse.py`); a published sign error in an LBM
wall-slip formula (arXiv:1603.09577 Eq. 73) found and documented
(`betaflow/analytic/lattice_boltzmann.py`); OpenLB's shipped
advection-diffusion benchmark shown to realise D_eff = 0.908 of its
requested diffusivity, predicted to 0.5% before measurement
(`results/openlb_first_contact.json`).

**Tier 1 — REPLICATE (one solver against the exact solution).** Each solver
takes the same exam separately: first a rung where the discrete answer is
known exactly (a null test, or exact kinematics with only statistical error),
then the physics rung. A solver never meets another solver here.

**Tier 2 — BENCHMARK (solver against solver, exact solution as referee).**
The same case runs through several runners with the same metrics, and each
solver's error is PREDICTED by its own analytic reference before it runs.
With three independent implementations, a later disagreement can be
localised rather than merely detected. The `mc_channel` case is the first
Tier-2 case: Langevin (green), OpenFOAM particles, and OpenLB.

## Architecture: four layers, one of which knows solvers exist

1. **Analytic references** (`betaflow/analytic/`) — exact solutions as pure
   functions with self-checks. No solver knowledge, no file I/O.
2. **Cases** (`betaflow/cases/*.yaml`) — declarative: geometry,
   non-dimensional groups, the dotted path of the reference (`reference:`
   key), named metrics with tolerances and their error laws.
3. **Runners** (`betaflow/runners/`) — adapters from cases to solvers, and
   the ONLY layer permitted to know a solver exists. The contract (from
   `runners/__init__.py`): each runner exposes `run(case: dict, **params)
   -> dict`, returning a `meta` provenance sub-dict plus whatever arrays the
   case's metrics consume. Six runners: `openfoam`, `openfoam_particles`,
   `langevin`, `moments`, `lbm`, and (planned) `openlb`.
4. **Metrics** (`betaflow/metrics/`) — consume plain arrays; looked up by
   the names used in case YAML. Statistical tolerances come from the error
   law of the specific estimator (`metrics/mc_error.py`), never from a
   round number.

Convergence is gated on conservation identities (relations the discrete
solution must satisfy exactly), never on iteration residuals — a residual
measures the size of the last step, an identity measures distance to the
solution. Every instrument is calibrated on a null case before being pointed
at an unknown.

## Case taxonomy

| Ladder | Case | Analytic reference | Runner(s) | Gate | Record |
|---|---|---|---|---|---|
| Fluid, steady | `couette_steady` | `analytic/couette.py` | openfoam | null test: round-off at every resolution | `results/couette_steady.json` |
| Fluid, steady | `poiseuille_steady` | `analytic/poiseuille.py` | openfoam | momentum identity tau_w = G V/A_wall | `results/poiseuille_steady.json`, `_refinement.json` |
| Fluid, steady | `pipe_poiseuille_steady` | `analytic/pipe.py` | openfoam (wedge) | momentum identity (lever arm a/2) | `results/pipe.json` |
| Fluid, steady | `pipe_poiseuille_io` | `analytic/pipe.py` | openfoam (wedge, inlet/outlet) | momentum identity; Stage-A pairing | `results/pipe_io_stage_a.json` |
| Fluid, pulsatile | `womersley_pulsatile` | `analytic/womersley.py` | openfoam | transient identity tau_w = h(G - d<u>/dt); asserted on wall shear, not profiles | `results/womersley_pulsatile.json` |
| Fluid, pulsatile | `pipe_womersley_pulsatile` | `analytic/womersley.py` (Bessel kernel) | openfoam (wedge) | same, pipe form | (kernel study: `results/kernel_misfit_operating_point.json`) |
| Rheology | `carreau_steady`, `womersley_carreau` | `analytic/carreau.py` | openfoam | momentum identity | `results/carreau_steady.json`, `womersley_carreau.json` |
| Rheology | `casson_steady`, `pipe_casson_steady` | `analytic/casson.py` | openfoam | conservation identity (residuals stall; see paper §5) | `results/casson_steady.json` |
| Particles | `langevin_free` | `analytic/brownian.py` | langevin, openfoam_particles | MSD isotropy; sigma-scaled laws | `results/langevin_free.json`, `_openfoam.json` |
| Particles | `taylor_aris` | `analytic/taylor_aris.py` | langevin, openfoam_particles | radial invariant P(r) = 2r/a^2 (KS) | `results/taylor_aris.json`, `_openfoam.json` |
| Eulerian scalar | `scalar_dispersion` | `analytic/advection_diffusion.py` | moments | declared no-axial-mesh scope | `results/scalar_dispersion.json` |
| LBM scalar | `lbm_scalar` | `analytic/lattice_boltzmann.py` | lbm | mass drift; slip zero at Lambda = 3/16 | `results/lbm_scalar.json` |
| Comms | `mc_channel` | `analytic/channel_impulse.py` | langevin (+ openfoam_particles, openlb planned) | radial invariant (KS); binomial error laws | `results/mc_channel.json`, `_departure.json` |

### The two geometry tracks and their fixed roles

The cases split into two geometry tracks, kept deliberately and with
distinct jobs:

- **The pipe track is the physics programme.** Everything nanoparticle,
  molecular-communications, and LBM happens in circular-pipe geometry,
  because blood vessels are cylinders and every published exact result the
  transport work stands on is cylindrical (Taylor-Aris, Decuzzi, Hofmann's
  duct, Liu's micro-vessel). All new physics cases land here.
- **The channel track is finished and stays as the calibration and
  discrimination layer.** The plane-channel cases calibrated the
  instruments (the Couette null, the identity checker) and now keep the
  geometry traps armed: the kernel blind spot, the three factor-of-2
  differences, and the wrong-kernel guard in `tests/test_pipe.py` all need
  a channel reference to exist. The cases are committed, fast, and stable,
  so keeping them costs nothing.

The failure mode this structure prevents is silent mixing: any channel
formula that wanders into pipe work must fail a definition-agreement check
(the Re convention, pinned to twelve digits), a conservation identity
(written against G a/2), or the planted wrong-kernel comparison — loudly,
instead of surviving into a result.

Solver-free studies with their own records: the kernel blind spot
(`kernel_discrimination_scaling.json`), Stage-A discriminators
(`stage_a_discriminators.json`), the production identity audit
(`production_identity_audit*.json`), the regularisation gap map
(`gapmap_numax_claim.json`), OpenLB first contact
(`openlb_first_contact.json`).

## Test tiers and CI

- **analytic** (27 tests, ~2 s, `pytest -m analytic`): self-checks of every
  analytic reference plus the identity-checker fixture null. No solver
  install. Gates every push.
- **default** (39 tests): adds the pure-Python runners and the OpenFOAM
  cases; needs OpenFOAM 14 locally.
- **slow** (7 tests): the long Monte Carlo and OpenFOAM-particle runs.

CI re-runs the suite and fails if any committed record moves beyond a
stated, measured tolerance (`tools/compare_results.py`, numeric comparison,
rtol 1e-3 / atol 2e-7 measured from CI artifacts).

## The correction policy

Corrections are recorded in place, never amended away. A withdrawn
conclusion stays in the history with the reason for its withdrawal, because
the record of failure is data about how errors present. Current examples on
record: the depleted-tail prediction for the `mc_channel` departure run
(wrong in direction; the measured structure is enhancement first, then
termination), the eigentime attribution for the termination clock
(withdrawn after the pre-registered Peclet sweep measured the layer-escape
scaling instead — a one-point match to a named constant that dissolved
under the sweep, `results/eigentime_pe_sweep.json`), the withdrawn
error-floor claim, and the 8x/16.3x discriminator correction.

## Out of scope, deliberately

- RBC-resolved blood (Liu et al. 2018's spectrin-link machinery): betaFlow
  verifies transport coefficients and wall schemes, and the dilute
  nanoparticle limit of such models, only.
- Bifurcating geometry: nothing branching exists yet; it is the month-3
  physics contribution of the LBM channel plan.
- Momentum-lattice LBM (fluid solving): the ADE lattice is covered; the
  fluid lattice and its bounce-back wall position are declared UNRESOLVED in
  `analytic/lattice_boltzmann.py` and wait for the OpenLB momentum rung.
