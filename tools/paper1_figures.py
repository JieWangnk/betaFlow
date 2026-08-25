#!/usr/bin/env python3
"""The two paper-1 figures the outline marks as missing, print quality.

  report/fig_two_act_tail.(pdf|png)
      The measured CIR against the flow-dominated model, all three
      receivers collapsed onto the model's master curve (time in units
      of each receiver's t2, CIR normalised by its peak — the model then
      collapses to ONE dashed curve), with the two acts annotated:
      enhancement above the model, then termination below it. Curves are
      regenerated deterministically from the validated Langevin runner at
      N = 100000 (higher than the record's 30000 for a smoother line —
      same physics, same seed; the record's assertions are unaffected).

  report/fig_isi_rate.(pdf|png)
      Left: worst-case ISI ratio against symbol interval, measured vs
      the window-truncated model, with the 10% threshold. Right: the
      achievable rate at ISI <= 10% and 20% — including where the model
      certifies NO rate. All numbers from results/comms_rate_metrics.json.

Sizes target an IEEE single column (3.5 in); PDFs are vector.
Run: python3 tools/paper1_figures.py   (~1 min)
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

import yaml  # noqa: E402

from betaflow.analytic import channel_impulse as ci  # noqa: E402
from betaflow.runners import run_case  # noqa: E402

INK, INK2, GRID = "#17160f", "#52514e", "#e5e3da"
S = ["#2a78d6", "#eb6834", "#1baf7a"]

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.linewidth": 0.7,
})


def _style(ax):
    ax.tick_params(colors=INK2, labelsize=7.5, width=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)


def two_act_tail():
    case = yaml.safe_load(
        (REPO / "betaflow" / "cases" / "mc_channel.yaml").read_text())
    res = run_case(case, runner="langevin", n_particles=100000, seed=1,
                   time_horizon_over_t2=12.0)

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    _style(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # The model's master curve (identical for every receiver once time is
    # in t2 units and the CIR is normalised by its peak).
    x = np.geomspace(0.55, 14.0, 400)
    d0, cx0, v0 = 750e-6, 100e-6, 1.5e-3
    t2_0 = ci.peak_time(v0, d0, cx0)
    master = ci.cir(x * t2_0, v0, d0, cx0) / ci.peak_value(d0, cx0)
    ax.plot(x, master, color=INK2, linewidth=1.0, linestyle=(0, (4, 2.5)),
            label="flow-dominated model", zorder=3)

    for rec, col in zip(res["receivers"], S):
        t = np.asarray(rec["t"]) / rec["t2"]
        y = np.asarray(rec["cir_measured"]) / rec["peak_value"]
        sel = (t > 0.55) & (y > 3e-4)
        ax.plot(t[sel], y[sel], color=col, linewidth=1.2,
                label=f"measured, {rec['dbar']*1e6:.0f} µm")

    ax.set_xlim(0.55, 14.0)
    ax.set_ylim(3e-3, 2.0)
    ax.set_xlabel(r"time / $t_2$", color=INK2)
    ax.set_ylabel("CIR / peak", color=INK2)

    # The two acts, annotated once each.
    ax.annotate("act 1: enhanced\n(up to 1.67×)", xy=(3.6, 0.42),
                xytext=(1.35, 0.062), color=INK, fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.annotate("act 2: terminated\n(model predicts $10^{-2}$)",
                xy=(7.6, 0.02), xytext=(5.0, 0.0045), color=INK,
                fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.legend(frameon=False, fontsize=7, labelcolor=INK, loc="upper right",
              handlelength=1.6)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(REPO / "report" / f"fig_two_act_tail.{ext}", dpi=300)
    plt.close(fig)
    print("written: report/fig_two_act_tail.[pdf|png]")


def isi_rate():
    rec = json.loads(
        (REPO / "results" / "comms_rate_metrics.json").read_text())

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.1, 2.9), gridspec_kw={"width_ratios": [1.35, 1],
                                               "wspace": 0.33})
    for ax in (ax1, ax2):
        _style(ax)

    # Left: ISI ratio vs symbol interval.
    ax1.set_yscale("log")
    for r, col in zip(rec["receivers"], S):
        ts = [row["t_symbol_over_t2"] for row in r["sweep"]]
        ax1.plot(ts, [row["isi_ratio_measured"] for row in r["sweep"]],
                 color=col, linewidth=1.2,
                 label=f"{r['dbar_um']:.0f} µm, measured")
        ax1.plot(ts, [row["isi_ratio_model_truncated"] for row in r["sweep"]],
                 color=col, linewidth=1.0, linestyle=(0, (4, 2.5)), alpha=0.8)
    ax1.axhline(0.1, color=INK2, linewidth=0.8)
    ax1.text(0.42, 0.115, "ISI = 10%", color=INK2, fontsize=7)
    ax1.plot([], [], color=INK2, linewidth=1.0, linestyle=(0, (4, 2.5)),
             label="model (truncated)")
    ax1.set_xlabel(r"symbol interval $T_s$ / $t_2$", color=INK2)
    ax1.set_ylabel("worst-case ISI / signal", color=INK2)
    ax1.legend(frameon=False, fontsize=6.5, labelcolor=INK, ncol=2,
               handlelength=1.5, columnspacing=0.9)

    # Right: achievable rate at the two thresholds; the model's missing
    # bars ARE the result, so their absence is labelled.
    groups = [(r["dbar_um"], r["max_rate_hz"]) for r in rec["receivers"]]
    xpos = np.arange(len(groups), dtype=float)
    w = 0.2
    for j, (theta, shade) in enumerate((("0.1", 1.0), ("0.2", 0.55))):
        for k, (kind, hatch) in enumerate((("measured", None),
                                           ("model_truncated", "////"))):
            vals = [g[1][theta][kind] for g in groups]
            xs = xpos + (j * 2 + k - 1.5) * w
            for xi, v in zip(xs, vals):
                if v is None:
                    ax2.text(xi, 0.035, "none", rotation=90, fontsize=6,
                             color=INK2, ha="center", va="bottom")
                else:
                    ax2.bar(xi, v, width=w * 0.92,
                            color=(S[0] if kind == "measured" else "none"),
                            edgecolor=(INK if kind != "measured" else "none"),
                            linewidth=0.7, hatch=hatch, alpha=shade,
                            zorder=3)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels([f"{g[0]:.0f} µm" for g in groups])
    ax2.set_ylabel("achievable rate  [bit/s]", color=INK2)
    ax2.set_ylim(0, 2.9)
    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(facecolor=S[0], label="measured (solid: ISI ≤ 10%,\n"
                                    "faded: ≤ 20%)"),
        Patch(facecolor="none", edgecolor=INK, hatch="////",
              label="model, truncated"),
    ], frameon=False, fontsize=6.5, labelcolor=INK, loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(bottom=0.18)   # keep the x-label clear of the edge
    for ext in ("pdf", "png"):
        fig.savefig(REPO / "report" / f"fig_isi_rate.{ext}", dpi=300)
    plt.close(fig)
    print("written: report/fig_isi_rate.[pdf|png]")


def wall_position():
    """report/fig_wall_position.(pdf|png): the wall-placement measurement.

    |a_eff - a| against resolution, log-log, from
    results/openlb_wall_position.json. Bounce-back decays with order
    ~1.4 and sits INSIDE the geometric wall; the Bouzidi control decays
    with order ~2.1 on identical runs, isolating the wall treatment.
    Shift is plotted in physical units (fractions of a) so the slopes
    ARE the observed orders.
    """
    rec = json.loads((REPO / "results" / "openlb_wall_position.json")
                     .read_text())
    rows = [r for r in rec["rows"] if r["tau"] == 0.53
            and not r["tag"].endswith("_t300")]
    series = {}
    for r in rows:
        series.setdefault(r["wall"], []).append(
            (r["N"], abs(r["wall_shift_dx"]) / r["N"]))

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    labels = {"bb": "bounce-back", "bouzidi": "Bouzidi"}
    orders = {k: rec["observed_order"][k] for k in ("bb", "bouzidi")}
    for (wall, pts), c in zip(sorted(series.items()), S):
        pts.sort()
        n = np.array([p[0] for p in pts], float)
        s = np.array([p[1] for p in pts], float)
        ax.loglog(n, s, "o-", color=c, lw=1.4, ms=4.5, label=labels[wall])
        # slope guide at the mean observed order, anchored at the last point
        p = float(np.mean(orders[wall]))
        gn = np.array([n[-2], n[-1] * 1.35])
        gs = s[-1] * (gn / n[-1]) ** (-p)
        ax.loglog(gn, gs * 1.35, ls="--", lw=0.9, color=c, alpha=0.6)
        ax.annotate(f"order {p:.1f}", (gn[-1], gs[-1] * 1.35),
                    fontsize=7, color=c, ha="right", va="bottom")
    ax.set_xlabel("cells per radius $N$", color=INK)
    ax.set_ylabel(r"$|a_{\mathrm{eff}} - a|\,/\,a$", color=INK)
    ax.set_xticks([21, 41, 81], ["21", "41", "81"])
    ax.minorticks_off()
    _style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(REPO / "report" / f"fig_wall_position.{ext}", dpi=300)
    plt.close(fig)
    print("written: report/fig_wall_position.[pdf|png]")


def schematic():
    """report/fig_schematic.(pdf|png): the channel geometry, print form.

    Same construction as tools/animate_mc_channel.py::schematic (which
    keeps the titled PNG for the README gallery), redrawn vector at IEEE
    column size with the caption carrying what the title said.
    """
    a, v_mean, cx = 200e-6, 1.5e-3, 100e-6
    dbar = (150e-6, 750e-6, 1550e-6)
    fig, ax = plt.subplots(figsize=(3.5, 1.55))
    mm, L, x0 = 1e3, 2.2e-3, 0.05e-3
    for ys in (a, -a):
        ax.plot([0, L * mm], [ys * mm, ys * mm], color=INK, lw=1.4)
    for yy in np.linspace(-0.85 * a, 0.85 * a, 7):
        u = 2 * v_mean * (1 - (yy / a) ** 2)
        ax.annotate("", xy=((x0 + u * 0.12) * mm, yy * mm),
                    xytext=(x0 * mm, yy * mm),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    ax.axvline(x0 * mm, color=INK, lw=0.8, ls=":")
    ax.text(x0 * mm, 1.18 * a * mm, "release", ha="center", fontsize=7,
            color=INK)
    for d, col in zip(dbar, S):
        lo, hi = (x0 + d - cx / 2) * mm, (x0 + d + cx / 2) * mm
        ax.axvspan(lo, hi, ymin=0.18, ymax=0.82, color=col, alpha=0.22)
        ax.text((lo + hi) / 2, -1.38 * a * mm,
                f"{d*1e6:.0f} " + r"$\mu$m", ha="center", fontsize=7,
                color=col, fontweight="bold")
    ax.set_xlim(-0.05, L * mm)
    ax.set_ylim(-1.62 * a * mm, 1.5 * a * mm)
    ax.set_xlabel("x  [mm]", color=INK, fontsize=8)
    ax.set_ylabel("y  [mm]", color=INK, fontsize=8)
    _style(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(REPO / "report" / f"fig_schematic.{ext}", dpi=300)
    plt.close(fig)
    print("written: report/fig_schematic.[pdf|png]")


def coupled():
    """report/fig_coupled.(pdf|png): the coupled model, print form.

    Same data as tools/plot_mc_channel_coupled.py (which keeps the wide
    titled PNG): the solved profile from
    _runs/mc_channel_fluid_res12/profile.csv and the CIR from
    _runs/mc_channel_openlb_res12_u0.04_coupled/cir.csv. Stacked panels
    at IEEE column width; skipped with a message if _runs/ is clean.
    """
    fluid = REPO / "_runs" / "mc_channel_fluid_res12" / "profile.csv"
    cirf = (REPO / "_runs" / "mc_channel_openlb_res12_u0.04_coupled"
            / "cir.csv")
    if not (fluid.exists() and cirf.exists()):
        print("fig_coupled SKIPPED: regenerate _runs/ via "
              "tests/test_openlb.py::test_mc_channel_openlb_coupled")
        return
    u_mean, cx = 1.5e-3, 100e-6
    dbar = (150e-6, 750e-6, 1550e-6)
    prof = np.loadtxt(fluid, delimiter=",", skiprows=1)
    cir = np.loadtxt(cirf, delimiter=",", skiprows=1)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.5, 3.3),
        gridspec_kw={"height_ratios": [1, 1.35], "hspace": 0.52})
    y, u = prof[:, 0], prof[:, 1] / (2.0 * u_mean)
    yy = np.linspace(-1, 1, 300)
    ax1.plot(yy, 1 - yy**2, color=INK2, lw=1.1, ls="--",
             label="exact parabola")
    ax1.plot(y[::6], u[::6], "o", color=S[0], ms=3.2,
             markeredgecolor="white", markeredgewidth=0.8,
             label="OpenLB, solved")
    ax1.set_xlabel("r / a", color=INK, fontsize=8)
    ax1.set_ylabel(r"$u / u_{\max}$", color=INK, fontsize=8)
    ax1.legend(frameon=False, fontsize=6.8, loc="lower center")

    t = cir[:, 0]
    tt = np.linspace(t[1], t[-1], 1200)
    for k, (d, col) in enumerate(zip(dbar, S)):
        ax2.plot(tt, ci.cir(tt, u_mean, d, cx), color=INK2, lw=0.9,
                 ls="--")
        ax2.plot(t, cir[:, 1 + k], color=col, lw=1.3,
                 label=f"{d*1e6:.0f} " + r"$\mu$m")
    ax2.plot([], [], color=INK2, lw=0.9, ls="--", label="model")
    ax2.set_xscale("log")
    ax2.set_xlim(t[1], t[-1])
    ax2.set_xlabel("time  [s]", color=INK, fontsize=8)
    ax2.set_ylabel("CIR", color=INK, fontsize=8)
    ax2.legend(frameon=False, fontsize=6.8, ncols=2)
    for ax in (ax1, ax2):
        _style(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(REPO / "report" / f"fig_coupled.{ext}", dpi=300)
    plt.close(fig)
    print("written: report/fig_coupled.[pdf|png]")


if __name__ == "__main__":
    two_act_tail()
    isi_rate()
    wall_position()
    schematic()
    coupled()
