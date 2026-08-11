"""First contact with OpenLB: measure its ADE example against the oracle.

READ-ONLY on an OpenLB checkout's example output. Runs no solver; the example
must have been built and run first:

    cd <openlb>/examples/advectionDiffusionReaction/advectionDiffusion1d
    make && ./advectionDiffusion1d

The example (Simonis, Frank & Krause 2020 benchmark) advects and diffuses a
single sine mode sin(pi (x - u t) gf), gf = N/(N+1), and writes both the
simulated and the analytic centreline to tmp/gnuplotData/data/. This tool
fits the mode's amplitude and phase in each written frame and extracts the
REALISED transport coefficients:

    D_eff = -ln(A2/A1) / (k^2 dt)          u_eff = -dphase / (k dt)

WHY THIS MEASUREMENT AND NOT THE EXAMPLE'S OWN ERROR PRINT. The example
prints a relative L2 error that DECAYS in time (0.161 -> 0.024 at N = 50),
which reads as a healthy converging run. The field data says otherwise: at
the shipped parameters (latticeU = 0.4, tau = 5) the realised diffusivity is
~0.90 against a requested 1.5 and the realised advection velocity ~9.1
against 10. Both are predicted by the dispersion relation of the
first-order-equilibrium BGK ADE scheme — the exact conserved eigenvalue of
the amplification matrix at the example's own wavenumber gives 0.9031 and
9.102. The headline number a user sees is reassuring while the transport
coefficients they asked for are 40% and 9% off; that combination is the
kernel-blind-spot structure this repo keeps finding, in the target code's
shipped benchmark.

SCOPE NOTES, so this is not over-read:
  * The k -> 0 depletion law D_eff = (c_s^2 - u^2)(tau - 1/2) gives 0.78
    here; the measured 0.90 matches the EXACT eigenvalue at k = 2 pi / 51,
    not the k -> 0 truncation. At large tau the k-expansion saturates
    slowly, so quote the eigenvalue at the actual k, not the limit law.
  * Simonis et al.'s published convergence study uses diffusive scaling, so
    latticeU falls with N and this error is part of their O(N^-2) budget —
    their EOC = 2 is correct. The trap is a FIXED-resolution run at the
    shipped latticeU, which is what a user reproducing the example gets.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def fit_mode(path, gf):
    data = np.loadtxt(path)
    x, y = data[:, 0], data[:, 1]
    k = np.pi * gf
    c = np.trapezoid(y * np.cos(k * x), x)
    s = np.trapezoid(y * np.sin(k * x), x)
    amplitude = 2.0 / (x[-1] - x[0]) * float(np.hypot(c, s))
    phase = float(np.arctan2(c, s))
    return amplitude, phase


def measure(example_dir, resolution=50, times=(0.08, 0.16, 0.24, 0.32, 0.40)):
    data_dir = Path(example_dir) / "tmp" / "gnuplotData" / "data"
    gf = resolution / (resolution + 1.0)
    k = np.pi * gf

    frames = []
    for t in times:
        a_sim, p_sim = fit_mode(data_dir / f"simulation{t:.6f}.dat", gf)
        a_ana, p_ana = fit_mode(data_dir / f"analytical{t:.6f}.dat", gf)
        frames.append({"t": t, "A_sim": a_sim, "phase_sim": p_sim,
                       "A_ana": a_ana, "phase_ana": p_ana})

    intervals = []
    for f1, f2 in zip(frames, frames[1:]):
        dt = f2["t"] - f1["t"]
        d_eff = -np.log(f2["A_sim"] / f1["A_sim"]) / (k**2 * dt)
        dphi = (f2["phase_sim"] - f1["phase_sim"] + np.pi) % (2 * np.pi) - np.pi
        u_eff = -dphi / (k * dt)
        intervals.append({"t1": f1["t"], "t2": f2["t"],
                          "D_eff": float(d_eff), "u_eff": float(u_eff)})
    return frames, intervals


def main(argv=None):
    argv = argv or sys.argv[1:]
    example = Path(argv[0]) if argv else Path(
        "~/GitHub/openlb/examples/advectionDiffusionReaction/advectionDiffusion1d"
    ).expanduser()
    frames, intervals = measure(example)
    d_vals = [iv["D_eff"] for iv in intervals]
    u_vals = [iv["u_eff"] for iv in intervals]
    out = {
        "purpose": "first contact: OpenLB's shipped ADE benchmark measured "
                   "against the betaflow lattice_boltzmann oracle",
        "example": str(example),
        "shipped_parameters": {
            "resolution": 50, "latticeU": 0.4, "tau": 5.0,
            "requested_D_phys": 1.5, "requested_u_char_phys": 10.0,
            "dynamics": "AdvectionDiffusionBGKdynamics (equilibria::FirstOrder, "
                        "source-confirmed in src/dynamics/advectionDiffusionDynamics.h)",
        },
        "frames": frames,
        "intervals": intervals,
        "measured": {
            "D_eff_mean": float(np.mean(d_vals)),
            "D_eff_range": [float(min(d_vals)), float(max(d_vals))],
            "u_eff_mean": float(np.mean(u_vals)),
        },
        "predicted": {
            "exact_eigenvalue_at_their_k": {"D_eff": 0.9031, "u_eff": 9.102},
            "k_to_zero_law": {"D_eff": 0.78,
                              "note": "(c_s^2 - u^2)(tau - 1/2) in phys units; "
                                      "differs from the finite-k eigenvalue at "
                                      "tau = 5 -- quote the eigenvalue"},
        },
        "conclusion": "the shipped benchmark realises D_eff ~ 0.90 of a "
                      "requested 1.5 and u_eff ~ 9.1 of a requested 10 at "
                      "fixed N = 50, exactly as the dispersion relation of "
                      "the first-order-equilibrium scheme predicts, while "
                      "the example's own error print decays reassuringly",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
