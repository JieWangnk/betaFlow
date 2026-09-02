#!/usr/bin/env python3
"""Junction visuals from the production run outputs (no solver re-run).

  report/junction_flow.png   the solved velocity field on the branch
                             plane z = 0: |u| as a single-hue sequential
                             map, the bore outline, receiver windows.
  report/junction_cir.png    the five-window impulse response: mother
                             control, both daughter mirror pairs, the
                             straight-pipe model at matched path
                             (dashed), and the P1 story - naive half vs
                             the velocity-corrected 0.63x.

Inputs: _runs/junction_flow_res12_t3/ufield.csv,
        _runs/junction_scalar_res12_u0.04/cir.csv,
        results/bifurcation_g4_control.json (the N0 straight baseline),
        results/bifurcation_junction.json.
Regenerate the inputs via tests/test_bifurcation_junction.py.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from betaflow.analytic import channel_impulse as ci  # noqa: E402

INK, INK2, GRID, BG = "#17160f", "#52514e", "#e5e3da", "#fcfcfb"
S = ["#2a78d6", "#eb6834", "#1baf7a"]

A = 200e-6
AD = A * 2.0 ** (-1.0 / 3.0)
UM = 1.5e-3
CX = 100e-6
X0, SJ = 10 * (A / 12), 350e-6
XJ = X0 + SJ
TH = np.deg2rad(30.0)


def flow_figure():
    d = np.loadtxt(REPO / "_runs" / "junction_flow_res12_t3" / "ufield.csv",
                   delimiter=",", skiprows=1)
    z0 = np.abs(d[:, 2]) < 1e-9
    x, y = d[z0, 0] * 1e3, d[z0, 1] * 1e3
    um = np.sqrt(d[z0, 3]**2 + d[z0, 4]**2 + d[z0, 5]**2) * 1e3  # mm/s

    fig, ax = plt.subplots(figsize=(9.2, 4.6), facecolor=BG)
    ax.set_facecolor(BG)
    sc = ax.scatter(x, y, c=um, s=4.5, marker="s", cmap="Blues",
                    vmin=0.0, vmax=3.05, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.015)
    cb.set_label("|u|  [mm/s]", color=INK2, fontsize=10)
    cb.ax.tick_params(colors=INK2, labelsize=9)
    # centrelines
    ax.plot([0, XJ * 1e3], [0, 0], color=INK2, lw=0.8, ls=":")
    for sgn in (+1, -1):
        L = 2.2e-3
        ax.plot([XJ * 1e3, (XJ + L*np.cos(TH)) * 1e3],
                [0, sgn * L*np.sin(TH) * 1e3], color=INK2, lw=0.8, ls=":")
    # windows (path 150 mother; 750, 1550 in each daughter)
    def draw_window(dbar, sgn, col):
        if dbar <= SJ:
            xc, yc, ang, aa = X0 + dbar, 0.0, 0.0, A
        else:
            s = dbar - SJ
            xc = XJ + s*np.cos(TH)
            yc = sgn * s*np.sin(TH)
            ang, aa = sgn * np.degrees(TH), AD
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D
        r = Rectangle(((xc - CX/2) * 1e3, (yc - aa) * 1e3), CX * 1e3,
                      2*aa * 1e3, fill=False, edgecolor=col, lw=1.6)
        r.set_transform(Affine2D().rotate_deg_around(
            xc * 1e3, yc * 1e3, ang) + ax.transData)
        ax.add_patch(r)
    draw_window(150e-6, +1, S[0])
    for sgn in (+1, -1):
        draw_window(750e-6, sgn, S[1])
        draw_window(1550e-6, sgn, S[2])
    ax.set_xlim(-0.05, 2.6)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.set_xlabel("x  [mm]", color=INK2, fontsize=10)
    ax.set_ylabel("y  [mm]", color=INK2, fontsize=10)
    ax.set_title("The solved junction flow (branch plane, 12 cells/radius)"
                 "  —  no recirculation, split symmetric to 0.8%",
                 color=INK, fontsize=11)
    ax.tick_params(colors=INK2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = REPO / "report" / "junction_flow.png"
    fig.savefig(out, dpi=170, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {out}")


def cir_figure():
    d = np.loadtxt(REPO / "_runs" / "junction_scalar_res12_u0.04" /
                   "cir.csv", delimiter=",", skiprows=1)
    t = d[:, 0]
    g4 = json.loads((REPO / "results" /
                     "bifurcation_g4_control.json").read_text())
    n0 = {r["dbar_um"]: r["peak_measured"]
          for r in g4["legs"]["N0"]["receivers"]}

    fig, ax = plt.subplots(figsize=(9.2, 5.0), facecolor=BG)
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.7)

    ax.plot(t, d[:, 1], color=S[0], lw=1.8, label="mother control, 150 µm")
    for col, kk, ls, lab in ((2, "750", "-", "daughter(+), 750 µm"),
                             (4, "750", (0, (5, 2)), "daughter(−), 750 µm"),
                             (3, "1550", "-", "daughter(+), 1550 µm"),
                             (5, "1550", (0, (5, 2)),
                              "daughter(−), 1550 µm")):
        c = S[1] if kk == "750" else S[2]
        ax.plot(t, d[:, col], color=c, lw=1.8, ls=ls, label=lab)
    # straight-pipe model at matched path, dashed ink
    for dbar in (150e-6, 750e-6, 1550e-6):
        tt = np.linspace(t[1], t[-1], 900)
        ax.plot(tt, ci.cir(tt, UM, dbar, CX), color=INK2, lw=1.0,
                ls=":", zorder=1)
    ax.plot([], [], color=INK2, lw=1.0, ls=":",
            label="straight-pipe model (matched path)")
    # the P1 story at 750 um
    half = 0.5 * n0[750.0]
    corr = half / (2.0 ** (-1.0 / 3.0))
    ax.axhline(half, color=S[1], lw=0.9, alpha=0.45)
    ax.axhline(corr, color=S[1], lw=0.9, alpha=0.85)
    ax.annotate("naive half of straight peak", (2.05, half),
                fontsize=8.5, color=S[1], alpha=0.8,
                va="bottom", ha="right")
    ax.annotate("velocity-corrected 0.63× (P1)", (2.05, corr),
                fontsize=8.5, color=S[1], va="bottom", ha="right")

    ax.set_xscale("log")
    ax.set_xlim(t[1], t[-1])
    ax.set_xlabel("time  [s]", color=INK2, fontsize=10)
    ax.set_ylabel("fraction inside window (CIR)", color=INK2, fontsize=10)
    ax.set_title("The junction impulse response — mirror pairs overlap "
                 "to 0.3%; daughters exceed the naive half by the "
                 "velocity factor", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncols=2)
    ax.tick_params(colors=INK2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = REPO / "report" / "junction_cir.png"
    fig.savefig(out, dpi=170, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {out}")


if __name__ == "__main__":
    flow_figure()
    cir_figure()
