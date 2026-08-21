# Paper 1 outline: a verified lattice-Boltzmann channel model for microfluidic molecular communications

**Target venue:** IEEE Transactions on Molecular, Biological and
Multi-Scale Communications (per the LBM channel plan). **Anchor paper:**
Hofmann et al. 2024, IEEE Access (OpenFOAM + experimental validation of
the same channel). **The gap:** LBM × molecular communications is a
zero-paper intersection (GAPS_AND_FRONTIERS survey); the competing
analytic line is the MIGHT/inverse-Gaussian family. Every claim below
names its committed record; the betaFlow methods paper is the citable
foundation for the verification approach and deliberately excludes the
LBM results, which are this paper's.

## The argument in one paragraph

Simulation is how molecular-communications channels beyond closed-form
reach get characterised, and the field's simulators are rarely verified
beyond profile-level agreement. We build a lattice-Boltzmann channel
model in OpenLB for the one experimentally validated microfluidic channel
in the literature, verify every layer of it against exact solutions —
transport coefficients, wall placement, the impulse response itself,
cross-checked by two independent particle implementations — and then use
the verified model to show that the standard flow-dominated analytic
model fails in a structured, measurable way that changes the
communications answers: its interference diverges, its memory is
undefined, and it understates the channel's achievable rate.

## Sections, claims, and evidence

### 1. Introduction
- MC channel characterisation needs simulators; verification practice is
  thin; Hofmann et al. provide the experimentally validated anchor.
- Why LBM for microfluidic channels (regime fit; the FV counterfactual:
  first-order upwind carries 0.5–10× the physical diffusion at any
  affordable mesh at Pe = 200 — `results/hofmann_validity_audit.json`).
- Contributions list (the five results below).

### 2. The channel and the verification ladder (methods)
- The channel: Table-1 pipe, Pe = 200; our dimensional split stated.
- The ladder: re-examine / replicate / benchmark; tolerances with stated
  origins; corrections kept in place. Cite the betaFlow methods paper;
  summarise only what this paper needs.
- The exact CIR re-derived independently (uniform-speed lemma), 23
  self-checks. `betaflow/analytic/channel_impulse.py`.

### 3. Result 1 — the lattice verified as a transport solver
- Depletion law D_eff = (c_s² − u²)(τ − ½) derived, then found live in
  OpenLB's own shipped benchmark (0.908 realised of 1.5 requested,
  predicted to 0.5%). `results/openlb_first_contact.json`.
- Stability pins τ against ½ at this Peclet on both lattices (measured
  divergence otherwise) — the sharp corner where only the eigenvalue law
  survives. `results/mc_channel_openlb.json`, `pipe_openlb.json`.
- Wall placement measured with a control: bounce-back's effective radius
  sits inside the geometric one, decaying as dx^1.4 (order 1.4–1.3);
  Bouzidi shift ∝ dx², order 2.1 — the control isolates the wall and
  fixes the design choice. `results/openlb_wall_position.json`.
- Scalar-wall slip zero at Λ = 3/16 measured to ten digits on the
  mini-lattice; published sign error documented.
  `results/lbm_scalar.json`.

### 4. Result 2 — the CIR benchmark: three solvers, one referee
- Langevin (exact-kinematics rung within binomial floors), OpenFOAM
  particles (the Hofmann model-class replication; MPPIC impossibility and
  DMPPIC deletion as citable negative findings), OpenLB Eulerian (+4.0%
  peak lag, tail above the particle legs by its scheme dispersion).
  `results/mc_channel_benchmark.json` and per-leg records.

### 5. Result 3 — where the analytic model fails, measured
- The two-act tail: enhancement (1.67× at 5 t₂; reservoir mechanism) then
  termination (zero at 12 t₂ against the model's 1e-2), cross-confirmed
  by two independent implementations. `results/mc_channel_departure.json`,
  `mc_channel_openfoam.json`.
- The validity clock: pre-registered Pe sweep refutes the eigentime
  attribution and measures t_cross = K·τ_r^0.31·dbar^0.73 — the
  layer-escape scaling; prefactor 2.78 (derivation open, stated).
  `results/eigentime_pe_sweep.json`. The refutation trail is presented as
  method, not embarrassment.

### 6. Result 4 — the communications consequences
- Under the model, worst-case ISI diverges logarithmically in the number
  of bits: channel memory undefined at every rate (closed form, self-
  checked). The measured termination makes memory finite (≈ 8 symbols at
  T_s = t₂, middle receiver).
- Achievable rate at ISI thresholds: the model certifies no rate at 10%
  for the near and far receivers where the measured channel supports
  1.93 and 0.37 Hz, and understates the middle receiver by 1.6×.
  `results/comms_rate_metrics.json`.

### 7. Result 5 — the coupled model (production configuration)
- OpenLB solves the flow it advects on (water, Re = 0.60, Sc = 667);
  staged coupling exact for steady flow, and the Sc-driven shared-clock
  trap stated; measured cost of the solved flow on the CIR: sub-percent
  peaks, tail ratios within 0.04. `results/mc_channel_openlb_coupled.json`.

### 8. Discussion and limits
- Steady flow, single straight pipe, one Pe decade for the clock sweep;
  the layer-escape O(1) constant underived; pulsatile coupling (the
  two-clock problem) and bifurcation geometry as the forward programme.
- What verification bought: every error component in the final model is
  separately measured; each wrong in-house prediction was caught by a
  pre-registered measurement and kept.

## Figures (assets exist unless marked)

1. Setup schematic — `report/mc_channel_schematic.png`.
2. The two-act tail: measured vs model CIR (from the coupled-figure tool;
   a publication-quality still of the animation's lower panel — NEEDS a
   dedicated version with all three receivers and both acts annotated).
3. Wall-position sweep, both wall treatments — Ledger chart 1 (rebuild at
   print quality from `results/openlb_wall_position.json`).
4. Crossover-clock sweep with refuted slope-1 guide — Ledger chart 2
   (rebuild likewise from `results/eigentime_pe_sweep.json`).
5. ISI ratio and achievable rate vs symbol interval, model vs measured —
   NEEDS building from `results/comms_rate_metrics.json`.
6. Coupled-model figure — `report/mc_channel_coupled.png`.

## Before submission (owner)

- Author block, affiliations, funding (author).
- Whether to cite Wicke et al. directly (currently attributed via
  Hofmann, flagged unread) — read or drop (author decision).
- Layer-escape constant: attempt the derivation or scope it out (joint).
- Pe sweep extension to a second decade if a reviewer-proof clock claim
  is wanted (compute, ~1 hour).
- Prose pass with the check-academic-prose skill; house language rules.
