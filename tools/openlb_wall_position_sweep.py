#!/usr/bin/env python3
"""Where does the bounce-back wall sit? The full sweep behind the record.

Measures the effective radius of OpenLB's D3Q19 pipe (steady forced
Poiseuille, openlb_cases/pipeFlow3d) across resolution, wall treatment,
and tau, and writes results/openlb_wall_position.json. The slow-tier
regression test (tests/test_openlb.py::test_pipe_momentum_openlb) re-runs
only the cheap (N = 21, 41) subset; THIS tool owns the expensive points —
N = 81 (~25 min each at tau = 0.53) and the tau probe.

Design notes, decided before the sweep and recorded here:
  - tau FIXED at 0.53 for the resolution sweep: bounce-back's wall
    position is tau-dependent, so a tau that varies with N (as a
    fixed-u_lat scaling would force) conflates the two dependences. At
    fixed tau the lattice velocity halves per refinement and stays stable.
  - The tau probe lives at N = 81 because stability ALLOWS a range only
    there (u_lat = (tau - 1/2) c_s^2 Re_cell; coarser grids pin tau
    harder against 1/2 — the same stability-pins-tau structure as the ADE
    leg, now on the momentum lattice).
  - A doubled-time pair (maxt 120 vs 300 at N = 21) bounds the
    convergence budget's contribution to the fitted shift.
  - Bouzidi (link-wise interpolated wall, second order) runs at every
    resolution as the CONTROL: it shares Ma^2 compressibility and the
    convergence budget with bounce-back, so the bb-minus-bouzidi
    difference isolates the wall.

DETERMINISTIC REUSE: the app has no randomness, so a config's output is a
pure function of its parameters. When a config's sweep directory already
holds a profile.csv, the tool reuses it instead of re-running, and says
so per-row (reused: true). Delete openlb_cases/pipeFlow3d/sweep_* to
force a clean re-measurement (~60 min).
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from betaflow.provenance import git_sha  # noqa: E402

APP_DIR = REPO / "openlb_cases" / "pipeFlow3d"
RADIUS = 1.0

# (tag, N, tau, wall, maxt) — the tags double as directory names.
CONFIGS = [
    ("bb21_t120", 21, 0.53, "bb", 120),
    ("bb21_t300", 21, 0.53, "bb", 300),   # convergence-budget pair
    ("bb41", 41, 0.53, "bb", 120),
    ("bb81", 81, 0.53, "bb", 120),
    ("bz21", 21, 0.53, "bouzidi", 120),
    ("bz41", 41, 0.53, "bouzidi", 120),
    ("bz81", 81, 0.53, "bouzidi", 120),
    ("bb81_tau0.58", 81, 0.58, "bb", 120),
    ("bb81_tau0.64", 81, 0.64, "bb", 120),
]


def fit(path, N):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    y, u = d[:, 0], d[:, 1]
    sel = np.abs(y) <= 0.85
    A = np.column_stack([np.ones(int(sel.sum())), y[sel] ** 2])
    c, *_ = np.linalg.lstsq(A, u[sel], rcond=None)
    a_eff = float(np.sqrt(-c[0] / c[1]))
    return {
        "u_max_fit": float(c[0]),
        "a_eff": a_eff,
        "wall_shift_dx": (a_eff - RADIUS) / (2.0 * RADIUS / N),
        "L2_vs_a_wall_parabola": float(np.sqrt(np.mean((u - (1 - y**2)) ** 2))),
    }


def main():
    if not (APP_DIR / "pipeFlow3d").is_file():
        subprocess.run(["make"], cwd=APP_DIR, check=True, capture_output=True)

    rows = []
    for tag, N, tau, wall, maxt in CONFIGS:
        out = APP_DIR / f"sweep_{tag}"
        csv = out / "profile.csv"
        reused = csv.is_file()
        if not reused:
            out.mkdir(exist_ok=True)
            p = subprocess.run(
                ["./pipeFlow3d", "--resolution", str(N), "--tau", repr(tau),
                 "--wall", wall, "--maxt", repr(float(maxt)),
                 "--outdir", f"sweep_{tag}/"],
                cwd=APP_DIR, capture_output=True, text=True)
            assert "betaflow-done" in p.stdout, p.stdout[-500:]
        row = {"tag": tag, "N": N, "tau": tau, "wall": wall,
               "maxt_s": maxt, "reused_existing_output": reused}
        row.update(fit(csv, N))
        rows.append(row)
        print(f"{tag:14s} shift {row['wall_shift_dx']:+.3f} dx  "
              f"L2 {row['L2_vs_a_wall_parabola']:.3e}"
              f"{'  (reused)' if reused else ''}")

    def order(tags):
        sel = [r for t in tags for r in rows if r["tag"] == t]
        return [float(np.log(c["L2_vs_a_wall_parabola"] / f["L2_vs_a_wall_parabola"])
                      / np.log(f["N"] / c["N"]))
                for c, f in zip(sel, sel[1:])]

    conv_pair = {r["tag"]: r["wall_shift_dx"] for r in rows
                 if r["tag"].startswith("bb21_t")}
    record = {
        "question": "effective wall position of bounce-back on a staircase "
                    "cylinder, D3Q19 forced Poiseuille",
        "app": "openlb_cases/pipeFlow3d/pipeFlow3d.cpp",
        "rows": rows,
        "observed_order": {
            "bb": order(["bb21_t120", "bb41", "bb81"]),
            "bouzidi": order(["bz21", "bz41", "bz81"]),
        },
        "convergence_budget_check": {
            "shift_at_maxt_120": conv_pair.get("bb21_t120"),
            "shift_at_maxt_300": conv_pair.get("bb21_t300"),
        },
        "tau_probe_N81_bb": [
            {"tau": r["tau"], "wall_shift_dx": r["wall_shift_dx"]}
            for r in rows if r["N"] == 81 and r["wall"] == "bb"],
        "git_sha": git_sha(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = REPO / "results" / "openlb_wall_position.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nbb order:      {record['observed_order']['bb']}")
    print(f"bouzidi order: {record['observed_order']['bouzidi']}")
    print(f"tau probe:     {record['tau_probe_N81_bb']}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
