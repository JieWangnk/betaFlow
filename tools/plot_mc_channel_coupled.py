#!/usr/bin/env python3
"""Two-panel figure of the coupled channel model, from run outputs.

Reads the files the coupled leg leaves in _runs/ (no solver re-run):
  _runs/mc_channel_fluid_res12/profile.csv                the SOLVED flow
  _runs/mc_channel_openlb_res12_u0.04_coupled/cir.csv     the CIR on it

Left panel: the solved velocity profile against the exact parabola — the
fluid stage's own exam, visually. Right panel: the three receivers'
channel impulse responses on the solved flow, against the flow-dominated
analytic model (dashed), so the two-act departure — depressed peak,
enhanced tail — is visible directly.

Regenerate the inputs first if _runs/ is clean:
  python3 -m pytest tests/test_openlb.py::test_mc_channel_openlb_coupled -q -m ""
Then:  python3 tools/plot_mc_channel_coupled.py    -> report/mc_channel_coupled.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from betaflow.analytic import channel_impulse as ci  # noqa: E402

# Palette (validated set): ink/greys + three categorical series.
INK, INK2, GRID = "#17160f", "#52514e", "#e5e3da"
S = ["#2a78d6", "#eb6834", "#1baf7a"]
BG = "#fcfcfb"

FLUID = REPO / "_runs" / "mc_channel_fluid_res12" / "profile.csv"
CIR = REPO / "_runs" / "mc_channel_openlb_res12_u0.04_coupled" / "cir.csv"
U_MEAN, CX = 1.5e-3, 100e-6
DBAR = (150e-6, 750e-6, 1550e-6)


def main():
    prof = np.loadtxt(FLUID, delimiter=",", skiprows=1)
    cir = np.loadtxt(CIR, delimiter=",", skiprows=1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.2), facecolor=BG,
        gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.28})
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.tick_params(colors=INK2, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK2)

    # Left: solved flow vs exact parabola.
    y, u = prof[:, 0], prof[:, 1] / (2.0 * U_MEAN)
    yy = np.linspace(-1, 1, 300)
    ax1.plot(yy, 1 - yy**2, color=INK2, linewidth=1.4, linestyle="--",
             label="exact parabola")
    ax1.plot(y[::6], u[::6], "o", color=S[0], markersize=4.5,
             markeredgecolor=BG, markeredgewidth=1.2, label="OpenLB, solved")
    ax1.set_xlabel("r / a", color=INK2, fontsize=10)
    ax1.set_ylabel("u / u_max", color=INK2, fontsize=10)
    ax1.set_title("The solved flow (Bouzidi walls, Re = 0.60)",
                  color=INK, fontsize=11)
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK)

    # Right: CIR on the solved flow vs the flow-dominated model.
    t = cir[:, 0]
    tt = np.linspace(t[1], t[-1], 1200)
    for k, (d, col) in enumerate(zip(DBAR, S)):
        ax2.plot(tt, ci.cir(tt, U_MEAN, d, CX), color=INK2, linewidth=1.1,
                 linestyle="--")
        ax2.plot(t, cir[:, 1 + k], color=col, linewidth=1.8,
                 label=f"receiver {d*1e6:.0f} µm")
    ax2.plot([], [], color=INK2, linewidth=1.1, linestyle="--",
             label="flow-dominated model")
    ax2.set_xscale("log")
    ax2.set_xlim(t[1], t[-1])
    ax2.set_xlabel("time  [s]", color=INK2, fontsize=10)
    ax2.set_ylabel("fraction inside receiver (CIR)", color=INK2, fontsize=10)
    ax2.set_title("Channel impulse response on the solved flow",
                  color=INK, fontsize=11)
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK)

    out = REPO / "report" / "mc_channel_coupled.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
