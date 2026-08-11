#!/usr/bin/env python3
"""Tier-0 audit of the anchor paper's analytic model, solver-free.

Answers two questions about Hofmann et al. 2024 (IEEE Access,
doi:10.1109/ACCESS.2024.3438243) from closed forms alone, and records the
numbers so the benchmark phases can cite them.

QUESTION 1 — how much of each receiver's tail can the flow-dominated model
describe at all? Past the crossover, the model's log-divergent tail
describes nothing: it first UNDERestimates inter-symbol interference
(enhancement regime) and then OVERestimates it (termination).

CORRECTION, RECORDED IN PLACE (11 Aug 2026). The first version of this
audit converted the crossover to peak-times through the radial relaxation
EIGENTIME tau_r/beta_1^2, on the strength of a single-point agreement
(measured 0.95 of it at Pe = 200, middle receiver). The pre-registered
Peclet sweep (tools/eigentime_pe_sweep.py,
results/eigentime_pe_sweep.json) REFUTED that clock: the measured scaling
is t_cross = K tau_r^0.31 dbar^0.73 — the layer-escape family
(exponents 1/3, 2/3), prefactor 2.78x the crude balance — and the
one-point eigentime match was a coincidence of the parameter point. The
eigentime-based extents (27.2 / 6.8 / 3.4 t2) are kept in this record as
the WITHDRAWN prediction; the measured extents at Pe = 200 are
7.9 / 6.4 / 5.5 t2, wrong in shape as well as size (the true extent
SHRINKS with receiver distance far more slowly than the eigentime form
said, and the near receiver's 27.2 was off by 3.4x).

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
        # WITHDRAWN eigentime form, kept per the correction policy:
        # 2 Pe a / (beta_1^2 (dbar + c_x/2)).
        withdrawn = 2.0 * peclet * a / (BETA_1**2 * (dbar + c_x / 2.0))
        # MEASURED law (results/eigentime_pe_sweep.json): the layer-escape
        # scaling with its measured prefactor, t_cross = 2.78 *
        # (a^2 dbar^2 / (32 D V^2))^(1/3), converted to this receiver's t2.
        t_cross = 2.78 * (a**2 * dbar**2
                          / (32.0 * DIFFUSIVITY * V_MEAN**2))**(1.0 / 3.0)
        receivers.append({
            "dbar_um": dbar * 1e6,
            "flow_dominated_ratio_paper": ci.flow_dominated(peclet, dbar, a),
            "t1_s_split_dependent": ci.onset_time(V_MEAN, dbar, c_x),
            "t2_s_split_dependent": t2,
            "model_valid_tail_extent_t2_units_measured": t_cross / t2,
            "model_valid_tail_extent_t2_units_WITHDRAWN_eigentime": withdrawn,
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
        "crossover_finding": (
            "the eigentime attribution is WITHDRAWN: the pre-registered "
            "Pe sweep (results/eigentime_pe_sweep.json) measured "
            "t_cross = K tau_r^0.31 dbar^0.73 — the layer-escape scaling "
            "(1/3, 2/3), prefactor 2.78x the crude balance; the one-point "
            "0.95-of-eigentime match at Pe = 200 was a coincidence"
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
              f"model tail extent {r['model_valid_tail_extent_t2_units_measured']:5.1f} t2 "
              f"(withdrawn eigentime form said "
              f"{r['model_valid_tail_extent_t2_units_WITHDRAWN_eigentime']:.1f})")
    for m in mesh_sweep:
        print(f"  dx = a/{m['cells_per_radius']:>3}: cell Pe {m['cell_peclet']:7.1f}, "
              f"D_num/D {m['d_num_over_d']:8.2f}, artefact {m['artefact_fraction']:.4f}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
