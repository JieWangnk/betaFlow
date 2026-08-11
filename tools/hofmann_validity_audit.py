#!/usr/bin/env python3
"""Tier-0 audit of the anchor paper's analytic model, solver-free.

Answers two questions about Hofmann et al. 2024 (IEEE Access,
doi:10.1109/ACCESS.2024.3438243) from closed forms alone, and records the
numbers so the benchmark phases can cite them.

QUESTION 1 — how much of each receiver's tail can the flow-dominated model
describe at all? The mc_channel departure measurement (commit 76cfc40,
results/mc_channel_departure.json) put the crossover where the measured CIR
leaves the model at ~0.95 tau_r/beta_1^2 — the radial relaxation eigentime,
an order of magnitude before tau_r itself. This audit converts that eigentime
into units of each receiver's own peak time t2: past roughly
(tau_r/beta_1^2)/t2 peak-times, the model's log-divergent tail describes
nothing, first UNDERestimating inter-symbol interference (enhancement
regime) and then OVERestimating it (termination). One seed, one parameter
point; the eigentime attribution is recorded as a hypothesis.

QUESTION 2 — what would an Eulerian finite-volume discretisation have
suffered at these parameters? Their published method is Lagrangian
(particle-based), and this quantifies the choice: at Pe = 200 with
first-order upwind advection, the scheme's own numerical diffusivity
D_num = (u dx/2)(1 - Co) rivals or exceeds the physical D at any affordable
mesh. The artefact fraction D_num/(D_num + D) is evaluated on a mesh sweep;
the same argument at haemodynamic parameters gave 0.99999
(betaflow/analytic/numerical_diffusion.py).

DIMENSIONAL SPLIT: Table 1 fixes Pe = V a / D = 200; the split
V = 1.5 mm/s, D = 1.5e-9 m^2/s is ours (cases/mc_channel.yaml). Every
Pe-based ratio below is the paper's; every absolute time depends on the
split and says so in its key.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from betaflow.analytic import channel_impulse as ci          # noqa: E402
from betaflow.analytic import numerical_diffusion as nd      # noqa: E402
from betaflow.provenance import git_sha                      # noqa: E402

BETA_1 = 3.8317059702  # first zero of J1: radial mode decays as exp(-beta_1^2 t/tau_r)

# Our dimensional split of the paper's Pe = 200 (cases/mc_channel.yaml).
V_MEAN = 1.5e-3
DIFFUSIVITY = 1.5e-9


def main():
    a = ci.HOFMANN_TABLE_1["radius"]
    c_x = ci.HOFMANN_TABLE_1["receiver_length"]
    peclet = ci.HOFMANN_TABLE_1["peclet"]
    tau_r = a**2 / DIFFUSIVITY
    eigentime = tau_r / BETA_1**2

    receivers = []
    for dbar in ci.HOFMANN_TABLE_1["mid_receiver_distances"]:
        t2 = ci.peak_time(V_MEAN, dbar, c_x)
        receivers.append({
            "dbar_um": dbar * 1e6,
            "flow_dominated_ratio_paper": ci.flow_dominated(peclet, dbar, a),
            "t1_s_split_dependent": ci.onset_time(V_MEAN, dbar, c_x),
            "t2_s_split_dependent": t2,
            # Pe-based, split-independent: eigentime/t2 = (a^2/D) / beta_1^2
            # / ((dbar + c_x/2)/(2V)) = 2 Pe a / (beta_1^2 (dbar + c_x/2)).
            "model_valid_tail_extent_t2_units":
                2.0 * peclet * a / (BETA_1**2 * (dbar + c_x / 2.0)),
            "peak_value": ci.peak_value(dbar, c_x),
        })

    # Eulerian counterfactual: first-order upwind at Co = 0.5, mesh sweep.
    courant_number = 0.5
    mesh_sweep = []
    for cells_per_radius in (5, 10, 25, 50, 100):
        dx = a / cells_per_radius
        mesh_sweep.append({
            "cells_per_radius": cells_per_radius,
            "dx_um": dx * 1e6,
            "cell_peclet": nd.cell_peclet(V_MEAN, dx, DIFFUSIVITY),
            "d_num_over_d": nd.numerical_diffusivity(
                V_MEAN, dx, courant_number) / DIFFUSIVITY,
            "artefact_fraction": nd.artefact_fraction(
                DIFFUSIVITY, V_MEAN, dx, courant_number),
        })

    record = {
        "paper": "Hofmann et al. 2024, doi:10.1109/ACCESS.2024.3438243",
        "peclet_paper": peclet,
        "dimensional_split_ours": {"v_mean": V_MEAN, "diffusivity": DIFFUSIVITY},
        "tau_r_s_split_dependent": tau_r,
        "eigentime_s_split_dependent": eigentime,
        "eigentime_over_tau_r": 1.0 / BETA_1**2,
        "crossover_hypothesis": (
            "measured CIR leaves the flow-dominated model at ~0.95 "
            "tau_r/beta_1^2 (one seed, one parameter point; "
            "results/mc_channel_departure.json)"
        ),
        "receivers": receivers,
        "eulerian_counterfactual": {
            "scheme": "first-order upwind, explicit, Co = 0.5",
            "statement": (
                "what an Eulerian FV discretisation would suffer at these "
                "parameters; their published method is Lagrangian and this "
                "quantifies that choice"
            ),
            "mesh_sweep": mesh_sweep,
        },
        "git_sha": git_sha(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = REPO / "results" / "hofmann_validity_audit.json"
    out.write_text(json.dumps(record, indent=2) + "\n")

    print(f"tau_r = {tau_r:.1f} s, eigentime tau_r/beta_1^2 = {eigentime:.2f} s")
    for r in receivers:
        print(f"  dbar {r['dbar_um']:6.0f} um: ratio {r['flow_dominated_ratio_paper']:5.1f}, "
              f"model tail extent {r['model_valid_tail_extent_t2_units']:6.1f} t2")
    for m in mesh_sweep:
        print(f"  dx = a/{m['cells_per_radius']:>3}: cell Pe {m['cell_peclet']:7.1f}, "
              f"D_num/D {m['d_num_over_d']:8.2f}, artefact {m['artefact_fraction']:.4f}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
