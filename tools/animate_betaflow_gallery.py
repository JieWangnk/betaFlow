#!/usr/bin/env python3
"""Animated gallery: one GIF per betaFlow story, from the exact physics.

Every animation uses either the analytic references directly or the
validated Langevin runner's own primitives (wall reflection, velocity
sampling), so the pictures illustrate the same dynamics the records
measure — at smaller particle counts, chosen for legibility. The records
remain the numbers of record; these are their moving pictures.

  report/gallery_langevin_free.gif   free Brownian motion + the 6Dt law
  report/gallery_taylor_aris.gif     pipe dispersion: stretch, then mix
  report/gallery_womersley_blindspot.gif
                                     pipe vs channel pulsatile kernels at
                                     alpha = 10: profiles indistinguishable,
                                     wall shear apart by half
  report/gallery_fv_artifact.gif     a pulse under first-order upwind vs
                                     the exact answer: the scheme's own
                                     diffusion, visible
  report/gallery_casson_plug.gif     tracer column in a Casson pipe: the
                                     plug rides as a solid block

Run: python3 tools/animate_betaflow_gallery.py   (~3-4 min, no solvers)
The companion mc_channel animation lives in tools/animate_mc_channel.py.
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

from betaflow.analytic import brownian, pipe, taylor_aris as ta  # noqa: E402
from betaflow.analytic import womersley as chan  # noqa: E402
from betaflow.runners.langevin import _specular_reflect  # noqa: E402

INK, INK2, GRID, BG = "#17160f", "#52514e", "#e5e3da", "#fcfcfb"
S = ["#2a78d6", "#eb6834", "#1baf7a"]
OUT = REPO / "report"


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=INK2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, draw, n_frames, name, fps=14):
    anim = FuncAnimation(fig, draw, frames=n_frames, blit=False)
    out = OUT / name
    anim.save(out, writer=PillowWriter(fps=fps), dpi=85)
    plt.close(fig)
    print(f"written: {out}")


# ---------------------------------------------------------------- free ----
def langevin_free():
    """Dots spread from the origin; the MSD grows on the 6Dt line."""
    D = brownian.stokes_einstein(310.0, 1e-3, 50e-9)   # 50 nm in plasma-ish
    n, steps, frame_every = 600, 900, 6
    dt = 1e-3
    rng = np.random.default_rng(0)
    x = np.zeros((n, 3))
    frames, msd, times = [], [], []
    for k in range(steps + 1):
        if k % frame_every == 0:
            frames.append(x[:, :2].copy())
            msd.append(float(np.mean(np.sum(x**2, axis=1))))
            times.append(k * dt)
        x += np.sqrt(2 * D * dt) * rng.standard_normal((n, 3))
    um = 1e6
    lim = 3.2 * np.sqrt(2 * D * steps * dt) * um

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), facecolor=BG,
                                   gridspec_kw={"wspace": 0.3})
    for ax in (ax1, ax2):
        _style(ax)
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x  [µm]", color=INK2, fontsize=10)
    ax1.set_ylabel("y  [µm]", color=INK2, fontsize=10)
    dots = ax1.scatter([], [], s=3.5, color=S[0], alpha=0.6, linewidths=0)
    title = ax1.set_title("", color=INK, fontsize=11, loc="left")
    ax2.grid(True, color=GRID, linewidth=0.7)
    tt = np.array(times)
    ax2.plot(tt, 6 * D * tt * um**2, color=INK2, linestyle="--",
             linewidth=1.3, label="6 D t (exact)")
    line, = ax2.plot([], [], color=S[0], linewidth=1.8, label="measured MSD")
    ax2.set_xlim(0, tt[-1]); ax2.set_ylim(0, 6 * D * tt[-1] * um**2 * 1.25)
    ax2.set_xlabel("time  [s]", color=INK2, fontsize=10)
    ax2.set_ylabel("mean squared displacement  [µm²]", color=INK2, fontsize=10)
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK)

    def draw(i):
        dots.set_offsets(frames[i] * um)
        line.set_data(times[:i + 1], np.array(msd[:i + 1]) * um**2)
        title.set_text(f"Free Brownian motion, 50 nm particle — t = {times[i]:.2f} s")
        return []
    _save(fig, draw, len(frames), "gallery_langevin_free.gif")


# --------------------------------------------------------- taylor-aris ----
def taylor_aris_gif():
    """Stretch, then mix: dispersion in the co-moving frame."""
    a = 10e-6
    u_mean = 3.1462e-6
    D = brownian.stokes_einstein(310.0, 1e-3, 150e-9)
    tau_r = a**2 / D
    n, t_end = 1500, 2.0 * tau_r
    dt = tau_r / 400.0
    steps = int(t_end / dt)
    frame_every = max(1, steps // 110)
    rng = np.random.default_rng(1)
    r0 = a * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    p = np.column_stack((r0 * np.cos(th), r0 * np.sin(th)))
    x = np.zeros(n)
    sig = np.sqrt(2 * D * dt)
    frames, var, times = [], [], []
    for k in range(steps + 1):
        if k % frame_every == 0:
            frames.append((x - u_mean * k * dt, p[:, 0].copy()))
            var.append(float(np.var(x)))
            times.append(k * dt)
        rr = np.hypot(p[:, 0], p[:, 1])
        x += ta.velocity_profile(rr / a, u_mean) * dt + sig * rng.standard_normal(n)
        p = _specular_reflect(p, p + sig * rng.standard_normal((n, 2)), a)

    um = 1e6
    d_eff = ta.d_eff(D, a, u_mean)
    span = 3.2 * np.sqrt(2 * d_eff * t_end) * um
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 5.6), facecolor=BG,
                                   gridspec_kw={"hspace": 0.35})
    for ax in (ax1, ax2):
        _style(ax)
    ax1.set_xlim(-span, span); ax1.set_ylim(-1.25 * a * um, 1.25 * a * um)
    for ys in (a, -a):
        ax1.axhline(ys * um, color=INK, linewidth=1.4)
    dots = ax1.scatter([], [], s=3, color=S[0], alpha=0.55, linewidths=0)
    ax1.set_xlabel("x − U t  [µm]  (co-moving frame)", color=INK2, fontsize=10)
    ax1.set_ylabel("y  [µm]", color=INK2, fontsize=10)
    title = ax1.set_title("", color=INK, fontsize=11, loc="left")
    ax2.grid(True, color=GRID, linewidth=0.7)
    tt = np.array(times)
    ax2.plot(tt / tau_r, (2 * d_eff * tt) * um**2, color=INK2, linewidth=1.3,
             linestyle="--", label="asymptotic slope 2 D_eff (exact)")
    line, = ax2.plot([], [], color=S[0], linewidth=1.8, label="measured variance")
    ax2.set_xlim(0, 2.0); ax2.set_ylim(0, 2 * d_eff * t_end * um**2 * 1.15)
    ax2.set_xlabel("t / tau_r", color=INK2, fontsize=10)
    ax2.set_ylabel("axial variance  [µm²]", color=INK2, fontsize=10)
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK)

    def draw(i):
        xs, ys = frames[i]
        dots.set_offsets(np.column_stack((xs * um, ys * um)))
        line.set_data(tt[:i + 1] / tau_r, np.array(var[:i + 1]) * um**2)
        title.set_text(f"Taylor–Aris dispersion, Pe = 21 — t = {times[i]/tau_r:.2f} tau_r")
        return []
    _save(fig, draw, len(frames), "gallery_taylor_aris.gif")


# ---------------------------------------------------- womersley kernels ---
def womersley_blindspot():
    """alpha = 10: two kernels one cannot tell apart, except at the wall."""
    alpha = 10.0
    r = np.linspace(-1, 1, 201)
    up = pipe.womersley_profile(np.abs(r), alpha)
    uc = chan.complex_profile(np.abs(r), alpha)
    tp = pipe.womersley_wall_shear(alpha)
    tc = chan.complex_wall_shear(alpha)
    n_frames = 84
    phases = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), facecolor=BG,
                                   gridspec_kw={"wspace": 0.3})
    for ax in (ax1, ax2):
        _style(ax)
    umax = 1.05 * float(np.max(np.abs(up)))
    ax1.set_xlim(-umax, umax); ax1.set_ylim(-1.02, 1.02)
    lp, = ax1.plot([], [], color=S[0], linewidth=2.0, label="pipe (J0 kernel)")
    lc, = ax1.plot([], [], color=S[1], linewidth=1.6, linestyle=(0, (4, 3)),
                   label="channel (cosh kernel)")
    ax1.axhline(1, color=INK, linewidth=1.2)
    ax1.axhline(-1, color=INK, linewidth=1.2)
    ax1.set_xlabel("velocity (normalised)", color=INK2, fontsize=10)
    ax1.set_ylabel("r / a", color=INK2, fontsize=10)
    ax1.set_title("Profiles: misfit 0.9%", color=INK, fontsize=11)
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax2.grid(True, color=GRID, linewidth=0.7)
    ph = np.linspace(0, 2 * np.pi, 300)
    wp = np.real(tp * np.exp(1j * ph))
    wc = np.real(tc * np.exp(1j * ph))
    ax2.plot(ph / (2 * np.pi), wp / np.max(np.abs(wp)), color=S[0], linewidth=1.8)
    ax2.plot(ph / (2 * np.pi), wc / np.max(np.abs(wp)), color=S[1],
             linewidth=1.6, linestyle=(0, (4, 3)))
    mark = ax2.axvline(0, color=INK2, linewidth=0.9, linestyle=":")
    ax2.set_xlabel("phase of the cycle", color=INK2, fontsize=10)
    ax2.set_ylabel("wall shear (normalised)", color=INK2, fontsize=10)
    ax2.set_title("Wall shear: misfit 48%", color=INK, fontsize=11)

    def draw(i):
        e = np.exp(1j * phases[i])
        lp.set_data(np.real(up * e), r)
        lc.set_data(np.real(uc * e), r)
        mark.set_xdata([phases[i] / (2 * np.pi)] * 2)
        return []
    _save(fig, draw, n_frames, "gallery_womersley_blindspot.gif", fps=18)


# -------------------------------------------------------- FV artefact -----
def fv_artifact():
    """First-order upwind vs the exact answer: the scheme's own diffusion."""
    nx, co = 220, 0.5
    x = (np.arange(nx) + 0.5) / nx
    c = np.exp(-((x - 0.15) / 0.03) ** 2)
    exact0 = c.copy()
    steps_total, frame_every = 660, 6
    frames, times = [], []
    cc = c.copy()
    for k in range(steps_total + 1):
        if k % frame_every == 0:
            # exact answer: the initial pulse advected without change
            frames.append((cc.copy(),
                           np.interp((x - co * k / nx) % 1.0, x, exact0)))
            times.append(k)
        cc = cc - co * (cc - np.roll(cc, 1))   # first-order upwind, Co = 0.5

    fig, ax = plt.subplots(figsize=(9.6, 3.8), facecolor=BG)
    _style(ax)
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.1)
    ax.grid(True, color=GRID, linewidth=0.7)
    le, = ax.plot([], [], color=INK2, linewidth=1.4, linestyle="--",
                  label="exact (pure advection)")
    lf, = ax.plot([], [], color=S[1], linewidth=1.9,
                  label="first-order upwind, Co = 0.5")
    ax.set_xlabel("x (periodic)", color=INK2, fontsize=10)
    ax.set_ylabel("concentration", color=INK2, fontsize=10)
    title = ax.set_title("", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    def draw(i):
        cc_i, ex_i = frames[i]
        lf.set_data(x, cc_i)
        le.set_data(x, ex_i)
        title.set_text("The scheme's own diffusion: D_num = u dx (1−Co)/2 "
                       f"— step {times[i]}")
        return []
    _save(fig, draw, len(frames), "gallery_fv_artifact.gif")


# --------------------------------------------------------- casson plug ----
def casson_plug():
    """Tracer columns: Newtonian shears everywhere; Casson's core is rigid."""
    xi = 0.2
    r = np.linspace(-0.98, 0.98, 41)
    u_n = pipe.poiseuille_profile(np.abs(r))
    u_c = pipe.casson_profile(np.abs(r), xi)
    u_c = u_c / np.max(u_c)
    n_frames, speed = 90, 0.012

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), facecolor=BG,
                             gridspec_kw={"wspace": 0.25})
    cols = []
    for ax, lbl in zip(axes, ("Newtonian", f"Casson (plug ratio {xi})")):
        _style(ax)
        ax.set_xlim(0, 1); ax.set_ylim(-1.05, 1.05)
        ax.axhline(1, color=INK, linewidth=1.4)
        ax.axhline(-1, color=INK, linewidth=1.4)
        ax.set_title(lbl, color=INK, fontsize=11)
        ax.set_xlabel("x (arbitrary)", color=INK2, fontsize=10)
        ax.set_xticks([])
    axes[0].set_ylabel("r / a", color=INK2, fontsize=10)
    for ax in axes[1:]:
        ax.axhline(xi, color=S[1], linewidth=0.9, linestyle=":")
        ax.axhline(-xi, color=S[1], linewidth=0.9, linestyle=":")
        ax.text(0.02, xi + 0.04, "plug edge r_p", color=S[1], fontsize=8)
    for ax, u in zip(axes, (u_n, u_c)):
        marks = [ax.plot([], [], "o", color=S[0], markersize=3.2,
                         linewidth=0)[0] for _ in range(4)]
        cols.append((marks, u))

    def draw(i):
        for marks, u in cols:
            for j, m in enumerate(marks):
                xpos = (0.12 + 0.22 * j + speed * i * u) % 1.0
                m.set_data(xpos, r)  # xpos is an ARRAY (one per r)
        return []
    _save(fig, draw, n_frames, "gallery_casson_plug.gif", fps=18)


if __name__ == "__main__":
    langevin_free()
    taylor_aris_gif()
    womersley_blindspot()
    fv_artifact()
    casson_plug()
