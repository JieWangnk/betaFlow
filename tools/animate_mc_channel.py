#!/usr/bin/env python3
"""Schematic and particle animation of the mc_channel setup.

Two outputs, both from the case's own physics (the wall reflection and
velocity sampling are imported from the validated Langevin runner, so the
pictures show the same dynamics the records measure):

  report/mc_channel_schematic.png   the setup, labelled: pipe, release
                                    plane, the three transparent receiver
                                    windows, the flow profile
  report/mc_channel_particles.gif   a side view of N = 4000 particles
                                    riding the flow at Pe = 200; a
                                    particle turns its receiver's colour
                                    while inside that window, and the
                                    lower panel grows the three CIR
                                    traces as the crowd moves — the
                                    "fraction inside receiver" curve IS
                                    the count of coloured dots.

Run time ~1 minute; no solver install needed.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from betaflow.analytic import taylor_aris as ta          # noqa: E402
from betaflow.runners.langevin import _specular_reflect  # noqa: E402

INK, INK2, GRID, BG = "#17160f", "#52514e", "#e5e3da", "#fcfcfb"
S = ["#2a78d6", "#eb6834", "#1baf7a"]

A, V, D, CX = 200e-6, 1.5e-3, 1.5e-9, 100e-6
DBAR = (150e-6, 750e-6, 1550e-6)
N, SEED = 4000, 1
T_END = 2.5
DT = ((DBAR[0] - CX / 2) / (2 * V)) / 25.0     # resolves the fastest rise
FRAME_EVERY = 15


def schematic():
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=BG)
    ax.set_facecolor(BG)
    mm = 1e3
    L = 2.2e-3
    # pipe walls
    for ys in (A, -A):
        ax.plot([0, L * mm], [ys * mm, ys * mm], color=INK, linewidth=2)
    # flow profile arrows at inlet
    for yy in np.linspace(-0.85 * A, 0.85 * A, 7):
        u = 2 * V * (1 - (yy / A) ** 2)
        ax.annotate("", xy=((0.05e-3 + u * 0.12) * mm, yy * mm),
                    xytext=(0.05e-3 * mm, yy * mm),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    # release plane
    ax.axvline(0.05e-3 * mm, color=INK, linewidth=1.2, linestyle=":")
    ax.text(0.05e-3 * mm, 1.15 * A * mm, "release\n(uniform puff, t = 0)",
            ha="center", fontsize=9, color=INK)
    # receiver windows
    for d, col in zip(DBAR, S):
        lo, hi = (0.05e-3 + d - CX / 2) * mm, (0.05e-3 + d + CX / 2) * mm
        ax.axvspan(lo, hi, ymin=0.18, ymax=0.82, color=col, alpha=0.20)
        ax.text((lo + hi) / 2, -1.32 * A * mm, f"receiver\n{d*1e6:.0f} µm",
                ha="center", fontsize=9, color=col, fontweight="bold")
    ax.text(1.95e-3 * mm, 0.55 * A * mm,
            "a = 200 µm   c_x = 100 µm\nPe = 200,  Re = 0.60",
            fontsize=9, color=INK2, ha="right")
    ax.set_xlim(-0.05, L * mm)
    ax.set_ylim(-1.6 * A * mm, 1.6 * A * mm)
    ax.set_xlabel("x  [mm]", color=INK2, fontsize=10)
    ax.set_ylabel("y  [mm]", color=INK2, fontsize=10)
    ax.set_title("The mc_channel setup (Hofmann Table 1)", color=INK,
                 fontsize=11)
    ax.tick_params(colors=INK2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    out = REPO / "report" / "mc_channel_schematic.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"written: {out}")


def animate():
    rng = np.random.default_rng(SEED)
    r0 = A * np.sqrt(rng.random(N))
    th = 2 * np.pi * rng.random(N)
    p = np.column_stack((r0 * np.cos(th), r0 * np.sin(th)))
    x = np.zeros(N)
    sig = np.sqrt(2 * D * DT)

    n_steps = int(np.ceil(T_END / DT))
    frames_x, frames_y, times, cirs = [], [], [], []
    cir_hist = [[], [], []]
    for k in range(n_steps + 1):
        if k % FRAME_EVERY == 0:
            frames_x.append(x.copy())
            frames_y.append(p[:, 0].copy())
            times.append(k * DT)
            for j, d in enumerate(DBAR):
                inside = (x >= d - CX / 2) & (x <= d + CX / 2)
                cir_hist[j].append(inside.mean())
            cirs.append([c[-1] for c in cir_hist])
        r_now = np.hypot(p[:, 0], p[:, 1])
        x += ta.velocity_profile(r_now / A, V) * DT + sig * rng.standard_normal(N)
        p = _specular_reflect(p, p + sig * rng.standard_normal((N, 2)), A)

    mm = 1e3
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), facecolor=BG,
        gridspec_kw={"height_ratios": [1.1, 1], "hspace": 0.32})
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.tick_params(colors=INK2, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    ax1.set_xlim(0, 4.0)
    ax1.set_ylim(-1.15 * A * mm, 1.15 * A * mm)
    ax1.set_xlabel("x  [mm]", color=INK2, fontsize=10)
    ax1.set_ylabel("y  [mm]", color=INK2, fontsize=10)
    for ys in (A, -A):
        ax1.axhline(ys * mm, color=INK, linewidth=1.6)
    for d, col in zip(DBAR, S):
        ax1.axvspan((d - CX / 2) * mm, (d + CX / 2) * mm, color=col, alpha=0.18)
    base = ax1.scatter([], [], s=2.5, color=INK2, alpha=0.45, linewidths=0)
    hots = [ax1.scatter([], [], s=6, color=c, linewidths=0) for c in S]
    title = ax1.set_title("", color=INK, fontsize=11, loc="left")

    ax2.set_xlim(0, T_END)
    ax2.set_ylim(0, 0.6)
    ax2.grid(True, color=GRID, linewidth=0.7)
    ax2.set_xlabel("time  [s]", color=INK2, fontsize=10)
    ax2.set_ylabel("fraction inside (CIR)", color=INK2, fontsize=10)
    lines = [ax2.plot([], [], color=c, linewidth=1.8,
                      label=f"{d*1e6:.0f} µm")[0] for d, c in zip(DBAR, S)]
    marker = ax2.axvline(0, color=INK2, linewidth=0.9, linestyle=":")
    ax2.legend(frameon=False, fontsize=9, ncol=3, labelcolor=INK)

    def draw(i):
        xf, yf, t = frames_x[i], frames_y[i], times[i]
        inside_any = np.zeros(len(xf), dtype=bool)
        for j, d in enumerate(DBAR):
            m = (xf >= d - CX / 2) & (xf <= d + CX / 2)
            hots[j].set_offsets(np.column_stack((xf[m] * mm, yf[m] * mm)))
            inside_any |= m
        base.set_offsets(np.column_stack((xf[~inside_any] * mm,
                                          yf[~inside_any] * mm)))
        for j, ln in enumerate(lines):
            ln.set_data(times[:i + 1], cir_hist[j][:i + 1])
        marker.set_xdata([t, t])
        title.set_text(f"N = {N} particles, Pe = 200 — t = {t:.2f} s")
        return []

    anim = FuncAnimation(fig, draw, frames=len(times), blit=False)
    out = REPO / "report" / "mc_channel_particles.gif"
    anim.save(out, writer=PillowWriter(fps=14), dpi=90)
    plt.close(fig)
    print(f"written: {out}")


if __name__ == "__main__":
    schematic()
    animate()
