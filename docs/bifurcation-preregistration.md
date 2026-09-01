# The bifurcation rung: pre-registration

Written 2026-08-26, before any junction geometry, solver case, or test
exists in this repository. The eigentime episode
(`results/eigentime_pe_sweep.json`) is the reason this document exists:
expectations written down before measurement are the only ones that count,
and wrong ones are kept beside their corrections. Nothing below changes
after the first junction run except by a dated correction in place.

## Why a bifurcation, and why now

The straight-channel programme is complete through the coupled model
(`report/RESULTS.md`): every solver error at the paper-1 operating point is
measured and bounded. A branching channel is the first geometry where no
closed-form impulse response exists at all, which makes it the plan's
physics contribution — and it is reachable now because the coupled model
needs no prescribed velocity profile, and because staging stays exact for
steady flow. The blocking prerequisite was the off-axis transport law: a
daughter branch runs oblique to the lattice, where the Ma² depletion was
conjecture until 2026-08-26. It is now the measured tensor law
(`results/lbm_offaxis_tensor.json`), and its budget appears as gate G6.

## The geometry (first case)

A planar, mirror-symmetric Y-junction, steady flow:

- Mother: the Table-1 pipe of the comms case — radius a = 200 µm, mean
  velocity V = 1.5 mm/s, water, Re = 0.60, Pe = 200, Sc = 667.
- Two identical daughters at ±30° to the mother axis (60° opening),
  radius from Murray's law for an equal split: a_d = a·2^(−1/3) = 158.7 µm.
- Murray consequences, fixed by mass conservation and geometry alone
  (each a checkable number, none of them free):
  mean daughter velocity V_d = 2^(−1/3) V = 1.19 mm/s;
  daughter Reynolds Re_d = 2^(−2/3) Re = 0.378;
  daughter Péclet Pe_d = 2^(−2/3) Pe = 126;
  daughter radial diffusion time τ_r,d = 2^(−2/3) τ_r = 16.8 s.
- Fluid lattice D3Q19 with Bouzidi walls; scalar lattice D3Q7 with
  bounce-back — the wall choices dictated by the wall-position measurement
  (`results/openlb_wall_position.json`).

Deliberately NOT decided here: mesh resolution (set later by the stability
pin at the realised daughter velocity), the geometry-construction route in
OpenLB, and receiver placement. Decisions made later are recorded in the
case file when made; expectations are what this document freezes.

## The gates

No exact junction solution exists, so the referee role passes to
conservation identities, measured-law budgets, and limiting cases. A gate
is never loosened to pass; a failure is diagnosed until its origin is
named.

**G1 — scalar mass.** Bulk-only accounting (bounce-back nodes hold no
readable density — the instrumentation trap recorded in
`results/mc_channel_openlb.json`). Retained-mass envelope: within the
straight channel's measured −2.7 % with the same mechanism named; any
excess over the straight-pipe envelope is a junction finding, to be
reported with its magnitude, never absorbed.

**G2 — fluid flux balance.** Q_mother = Q_d1 + Q_d2. Origin of the
tolerance: the coupled fluid exam's convergence floor (profile L2
3.5e-3 at 24 cells per diameter). Gate: relative imbalance ≤ 5e-3 at that
resolution, and it must SHRINK under refinement at the Bouzidi order
(~2, measured 2.1). An imbalance flat under refinement is an accounting
bug, not discretisation (the identity-pinned versus discretisation-borne
distinction of the pipe work).

**G3 — symmetry.** The geometry is mirror-symmetric, so for the
deterministic Eulerian legs Q_d1/Q_d2 = 1 and the two daughter CIRs are
identical to round-off — PROVIDED the discrete geometry actually has the
mirror plane (cell-alignment parity is the trap; a measured asymmetry
means broken alignment, a bug class, before it means physics). For
particle legs the split is binomial: 50/50 within √(0.25/N).

**G4 — the blocked-daughter limit.** Close one daughter: the case must
reproduce the straight-pipe behaviour it descends from. The open-path CIR
against the straight coupled record
(`results/mc_channel_openlb_coupled.json`) within that record's own
leg-vs-leg envelope (peaks to 0.4 %, tail ratios to 0.04), plus whatever
bend-specific deviation the 30° elbow itself causes — reported, not
absorbed. This is the strongest regression anchor and runs FIRST.

**G5 — wall placement.** Bouzidi fluid walls: convergence order ~2
expected (measured 2.1 on the cylinder). Scalar bounce-back: a radius bias
decaying as dx^1.4 (measured law), budgeted per daughter radius, never
tuned away.

**G6 — the off-axis tensor budget.** Daughter axes run at 30° to the
lattice. From the measured tensor law: cross-diffusion
D_xy = −(τ−½)u_x u_y with u_x u_y = |u|² sin30°cos30° = 0.433|u|². At the
straight case's u_lat ≈ 0.04 the relative anisotropy is
u_x u_y / c_s² ≈ 2.8e-3 of D — below G2's floor, and NOT removable by the
second-order equilibrium (measured: the cross term survives the fix at
full size on D3Q7-class sets). If daughter u_lat rises above ~0.1 in any
later configuration this budget line crosses a percent and must be
re-evaluated before that run.

**G7 — grid convergence.** Pre-registered expected orders: fluid
quantities ~2 (Bouzidi); any geometry-bias-controlled quantity 1.4
(bounce-back staircase); scalar peak time and peak value to be measured
and recorded with no order promised (the pinned τ = 0.5048 sits at the
scheme's sharpest corner, and promising an order there would be
guessing).

## Pre-registered physics expectations

Written before any run, with the named alternatives that could produce
the same numbers (`name-the-alternative` rule):

**P1 — peak split.** Each daughter receiver sees a peak of ~1/2 the
equivalent straight-pipe peak (symmetric advective split). Alternative
that mimics it: mass parked at the junction apex plus tail enhancement
could depress peaks below 1/2 while conserving mass — distinguished by G1
plus the tail integral.
*Measured 2026-09-01 (`results/bifurcation_junction.json`): the naive
half is WRONG by the velocity factor — daughter peaks run 1.23×/1.28×
half the same-instrument straight peak at 750/1550 µm. The corrected
prediction is 1/(2·2^{−1/3}) = 0.63 of the straight peak (the slower
Murray daughters concentrate the pulse in space by 1/V), which lands
within 3%; the peak arrival times likewise match the two-segment
kinematics (path/V stretched by the daughter speed, times the known +4%
lattice lag) at both receivers. The naive picture is kept beside its
correction per policy.*

**P2 — no recirculation.** At Re = 0.378 the junction flow is
Stokes-like; no recirculation bubble and no secondary-flow redistribution
of the slug is expected. Refutation would look like: a persistent local
minimum in the apex-region velocity magnitude, or daughter CIR shapes
that differ from a bent-pipe control. If refuted, that is a finding to
report (it would be the first junction-specific transport physics of the
programme).

**P3 — the crossover clock in the daughters.** The measured law
t_cross = K·τ_r^0.31·d̄^0.71 (18-point fit, Pe 50–3200) predicts the
daughters' validity clock shrinks by 2^(−2/3·0.31) = 0.87× at matched
receiver distance. This is the law's first out-of-family test: it was fit
on straight pipes, and a junction daughter is a straight pipe only after
an entrance length. Alternative: junction-induced radial mixing resets
the near-wall reservoir, shortening t_cross beyond the 0.87× — the two are
distinguished by sweeping receiver distance within one daughter.

**P4 — dispersion beyond closed form.** The junction adds a spreading
mechanism no straight-duct model carries: path-length dispersal at the
apex. Expected signature: daughter CIR wider than the straight-pipe CIR
at matched (Pe_d, d̄) even after the 1/2 scaling. No magnitude is
pre-registered — there is no committed law to compute one from, and
inventing one here would be the eigentime mistake again.

## Addendum 2026-08-26 — the oblique wall instability, and the bend numerics

This document deferred numerics to "the stability pin at the realised
daughter velocity". The G4 control forced the decision on its first day,
and the diagnosis went through four wrong-or-partial hypotheses before the
discriminating experiment isolated it — all kept here.

**The finding.** Stock BGK ADE (first-order equilibrium, D3Q7, bounce-back
walls) at the stability-pinned τ is UNSTABLE whenever the advection runs
oblique to the lattice. The discriminating ladder, all at Pe = 200:

| configuration | outcome |
|---|---|
| straight, axis-aligned, BGK τ = 0.5048 | stable (the committed record) |
| mitred 30° elbow, BGK τ = 0.5048 | diverges (mass −1.9e51) |
| mitred 30° elbow, BGK τ = 0.5096 (doubled margin) | diverges (−5.8e46) — damping is the wrong knob |
| smooth-arc 30° elbow, BGK | diverges — the kink was not the mechanism |
| straight OBLIQUE pipe (no bend at all), BGK | **diverges** — the wall is the mechanism |
| straight oblique, BGK second-order equilibrium | diverges |
| straight oblique, TRT τ_even = 1 | diverges — even-sector damping is not the knob |
| oblique, TRT Λ = 3/16 or 1/4, 6 cells/radius | no blowup over the horizon (mass 0.67) |
| oblique, TRT Λ = 1/4, **12 cells/radius** | bulk mass crosses **zero** at 3.27 s of 3.44 — the gain is orders lower, the loop persists; the res-6 "stability" was a rate effect (fewer steps per physical time) |
| plain Bouzidi reflection as the scalar wall, axis-aligned | stable but retains 0.25 of the scalar — the interpolation is not mass-conserving for a scalar (no pressure field to self-correct); **refuted** as the no-flux candidate |
| plain Bouzidi, oblique | leaks to 0.24, then diverges |
| Noble–Torczynski walls + BGK, oblique | diverges — the solid operator alone is not the cure |
| **Noble–Torczynski walls + TRT Λ = 1/4**, oblique, 12 cells/radius | **stable**: mass settles at ~0.88 (cut-cell transit share, decelerating — the opposite signature of the TRT+bounce-back drain), peaks physical |

Plane-wave stability of the uniform scheme is clean at every operating
point tested (max |λ| < 1, full 3-D wavevector scan; re-derived inside the
G4 test), so the instability lives in the wall interaction: oblique flow
puts nonzero advective flux through the staircase's bounce-back faces —
exactly zero in the axis-aligned case, which is why the straight programme
never saw it — and the barely-relaxed odd sector (ω_odd → 2 at the pin)
feeds the reflected populations back with near-unit gain.

**The scheme study (2026-08-26, `results/oblique_wall_scheme_study.json`).**
The wall-scheme question moved to the pure-numpy reference lattice — a
doubly periodic oblique channel with rational slope 1/2, where uniform
concentration under a Poiseuille profile is an exact steady state, so any
deviation is the wall scheme's own artifact. Measured there:

- **Bounce-back**: conservative; bounded in 2-D with an O(1) steady wall
  layer, amplitude ≈ 5 u_lat decaying over 2–3 cells (the same mechanism
  that diverges in 3-D D3Q7).
- **Non-equilibrium reflection** (the Zou–He-type scalar wall): exact on
  uniform C with uniform u — and REFUTED in general: its per-link
  exchange is proportional to local C, a positive-feedback pump (mass
  ×178 in 6000 steps on the profile null).
- **Noble–Torczynski partially-saturated cells** (u_w = 0 form): exactly
  conservative, stable, true-wall geometry, keeps the field positive
  where bounce-back undershoots, wall layer ~20% below bounce-back's —
  the best conservative candidate measured.
- **Layer scaling**: amp ≈ 5 u_lat, falling in (τ−½) with exponent ~−0.7.
  At the stability pin, refinement at fixed u_lat grows τ−½ linearly, so
  the layer shrinks ≈ res^0.7 — a ~1% window bias needs ~48 cells per
  radius.
- **No TRT Λ cures the layer** (amplitude grows with Λ); Λ = 1/12
  confines the mid-channel spill 4× below BGK.

Two costed paths forward: (a) Noble–Torczynski walls + Λ = 1/12 +
~48 cells/radius on HPC, the layer budgeted by the measured scaling —
needs the NT ADE operator ported to OpenLB; (b) a boundary-layer-
corrected wall scheme — open theory: the layer is the equilibrium
face-flux inconsistency, no collision knob removes it, and every
conservative local correction measured so far either keeps it or
destabilises.

**Decision.** The angle-0 control runs TRT at magic Λ = 1/4 (Ginzburg's
bounce-back stability optimum; the diffusivity rides the TRT's ODD rate —
measured on the exact dispersion relation — so Λ touches stability only)
and is gated on the pre-registered envelope against the straight BGK
record, covering the machinery and the collision change together. The
BEND leg is **blocked**, and that is gate G4's result: no wall treatment
available in stock OpenLB 1.9 gives stable impermeable-wall scalar
transport oblique to the lattice at these parameters — bounce-back feeds
the loop, TRT at magic Λ only slows it, and the shipped interpolated ADE
wall (`setBouzidiAdeDirichlet`) is absorbing, which is the wrong physics
for this channel. **The junction itself (2026-09-01, `results/bifurcation_junction.json`).**
All gates green at production resolution: G2 flux balance +0.14%;
G5/G7 profiles redeveloped (mother 3.8e-3, daughters ~1e-2); P2
confirmed — no recirculation, daughter centrelines at 99.2% of full
speed; scalar mirror windows within 0.3% relative. G3's bring-up
finding matured into a structural fact: the antisymmetric flux-split
mode is FLAT under refinement (0.72% at res 6 → 0.78% at res 12 —
the identity-vs-discretisation doctrine reads flat as structural, and
the leading candidate is the oblique outlet discs' boundary-normal
classification; bounded by gates, tracked, not absorbed). P3 awaits its
receiver-distance sweep. P4's inputs (tail metrics per window) are in
the record.

**Resolved 2026-08-27: the working wall is Noble–Torczynski
partially-saturated cells (exactly conservative, true-surface geometry)
with a TRT bulk at magic Λ = 1/4 — either ingredient alone fails.** The
G4 record carries the four-leg design: A0 (BGK machinery gate, 0.007%),
A1 (TRT collision calibration), N0 (the NT instrument's own calibration
against the record — TRT-sharpened peaks, tail ratios reading low because
near-wall tail content sits partly in cut cells under bulk-only
accounting), and N30 — **the first bend measurement**, quoted against N0
on the same instrument: the upstream in-run control window is unmoved
(gated), the post-bend window's peak −8.2%, the far window's +5.8%, with
the reference-lattice wall-layer budget (amp ≈ 5 u_lat over 2–3 cells,
shrinking ≈ res^0.7) riding with the numbers. Path (b) — the theory of a
layer-free closure — continues in
`results/halfspace_closure_study.json`.

**The solved-flow bend (2026-09-01).** The fluid stage runs through the
same elbow (bentFlow3d: D3Q19, Bouzidi walls on the shared bore surface,
inlet Poiseuille, outlet pressure, warm-started from the analytic field)
and passes its pre-registered gates at production resolution: flux
imbalance 7.8e-4 (G2's 5e-3 with a 6× margin), profile L2 ≈ 5e-3 at both
stations, and **P2 confirmed — the bend centreline holds 0.998 of the
full speed, no recirculation**. Two instrumentation findings joined the
trail: the steady LB pipe carries an O(10%) compressibility density
head at diffusive scaling (measured 1.048 at res 12, matching Δp/c_s²),
so the physical field is ρu normalised by the measured inlet-section
density; and a half-cell grid-convention mismatch in the field hand-off
once fed the scalar a silently-zero field (lag +51 t₂) — the scalar app
now aborts unless the loaded field carries the centreline speed at the
slug. The solved-vs-prescribed cost on the bend: peaks within 0.9%,
tails within 0.032, at all three receivers — the bend measurements
stand on the real flow.

## Order of work (when the rung starts)

1. G4 first: the bent-pipe / blocked-daughter control against the straight
   record — the instrument check before the measurement.
2. Fluid-only junction: G2, G3, G5, G7 on the velocity field.
3. Coupled scalar: G1, G6, then P1–P4.
4. Every record under `results/`, every wrong expectation kept in place.
