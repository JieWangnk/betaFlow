#!/usr/bin/env python3
"""The oblique no-flux wall problem, characterised on the reference lattice.

WHY THIS STUDY EXISTS. Gate G4 of the bifurcation pre-registration found
stock BGK ADE unstable whenever advection runs oblique to the lattice
(results/bifurcation_g4_control.json), and refuted plain Bouzidi
reflection as the interpolated no-flux candidate (not mass-conserving for
a scalar). This study takes the scheme question to the pure-numpy
reference lattice, where a candidate costs minutes instead of a C++
port: a doubly periodic oblique channel with rational slope
tan(theta) = 1/2 (phi = -x + 2y is integer on the lattice, so the
staircase is exactly periodic), D2Q5, prescribed flow along the channel,
and a null test with a known exact answer — uniform concentration with
a Poiseuille profile across the channel is a steady state of the
continuous problem, so ANY deviation is the wall scheme's own artifact.

WHAT IT MEASURES (all persisted in the record):

1. Bounce-back: exactly conservative; bounded here (2-D D2Q5) with an
   O(1) steady wall layer — amp ~ 5 u_lat at the wall, decaying over
   ~2-3 cells. (In 3-D D3Q7 at the pinned tau the same wall DIVERGES —
   the G4 record; the dimension difference is a caveat, not a
   contradiction: the layer mechanism is the same, the 3-D loop gain
   crosses 1.)
2. Non-equilibrium reflection (the Zou-He-type scalar wall,
   g_ibar = g_ibar^eq + (g_i* - g_i^eq)): EXACT on uniform C with
   uniform u (machine-zero deviation, mass conserved) — and REFUTED in
   general: its per-link exchange is proportional to the local C, a
   positive-feedback pump that diverges with any concentration or
   velocity gradient. Kept as the sharpest example of why wall
   exchanges proportional to local fields are forbidden.
3. Noble-Torczynski partially-saturated cells (solid fraction from the
   TRUE wall, pairwise-antisymmetric solid operator with u_w = 0):
   exactly conservative, stable, resolves the true wall — and carries
   the SAME wall layer as bounce-back, ~20% smaller. The best
   conservative candidate measured.
4. Layer scaling: amplitude ~ 5 u_lat and decreasing in (tau - 1/2)
   (measured exponent ~ -0.7 over tau 0.5048-0.53). At the stability
   pin, refinement at fixed u_lat GROWS tau - 1/2 linearly, so the
   layer shrinks ~ res^0.7: reaching a ~1% window bias needs roughly
   48 cells per radius.
5. No TRT magic Lambda cures the layer (amplitude GROWS with Lambda);
   Lambda = 1/12 — the advection magic — minimises the mid-channel
   spill (4x below BGK) by confining the layer.

Run: python3 tools/oblique_wall_scheme_study.py     (~10 min)
Writes: results/oblique_wall_scheme_study.json
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from betaflow.provenance import git_sha  # noqa: E402

P, W = 50, 30
LX = LY = 100
CS2 = 1.0 / 3.0
VEL = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
WGT = np.array([1.0/3.0] + [1.0/6.0]*4)
OPP = [0, 2, 1, 4, 3]

y, x = np.mgrid[0:LY, 0:LX]
phi = (-x + 2*y) % P
channel = phi < W
wall = ~channel
theta = np.arctan(0.5)
d_signed = ((phi - W/2.0 + P/2.0) % P) - P/2.0
dist = np.minimum(phi, np.clip(W - phi, 0, None)).astype(float) / np.sqrt(5.0)

_ss = np.linspace(-0.375, 0.375, 4)
eps = np.zeros((LY, LX))
for _dy in _ss:
    for _dx in _ss:
        eps += ((-(x + _dx) + 2*(y + _dy)) % P) >= W
eps /= 16.0
interior = eps == 0.0
notsolid = eps < 1.0


def _velocity(u0, profile):
    fac = (np.clip(1.0 - (d_signed/(W/2.0))**2, 0.0, None)
           if profile else np.ones_like(d_signed, float))
    return u0*np.cos(theta)*fac, u0*np.sin(theta)*fac


def _feq(C, ux, uy):
    out = np.empty((5,) + C.shape)
    for i, (cx, cy) in enumerate(VEL):
        out[i] = WGT[i] * C * (1.0 + (cx*ux + cy*uy)/CS2)
    return out


def _init(kind, ux, uy, mask):
    if kind == "uniform":
        C0 = np.where(mask, 1.0, 0.0)
    else:
        s = (2*x + y).astype(float)
        C0 = np.where(mask, np.exp(-((d_signed/6.0)**2)
                                   - (((s - 150.0)/30.0)**2)), 0.0)
    return _feq(C0, ux, uy)


def run_link_wall(scheme, tau, u0, steps, profile, init="uniform",
                  lam=None):
    """BB or NER walls on the staircase; BGK, or TRT when lam is given
    (odd rate carries D at tau; Lambda sets the even rate)."""
    ux, uy = _velocity(u0, profile)
    g = _init(init, ux, uy, channel)
    g[:, wall] = 0.0
    adj = {i: channel & np.roll(np.roll(wall, -cy, 0), -cx, 1)
           for i, (cx, cy) in enumerate(VEL) if i}
    if lam is not None:
        tau_even = lam/(tau - 0.5) + 0.5
        om_e, om_o = 1.0/tau_even, 1.0/tau
    m0 = g.sum()
    for t in range(steps):
        C = g.sum(axis=0)
        ge = _feq(C, ux, uy)
        if lam is None:
            g += -(g - ge)/tau
        else:
            gp, gm = 0.5*(g + g[OPP]), 0.5*(g - g[OPP])
            gep, gem = 0.5*(ge + ge[OPP]), 0.5*(ge - ge[OPP])
            g += -om_e*(gp - gep) - om_o*(gm - gem)
        rets = {}
        for i in adj:
            io = OPP[i]
            rets[i] = (g[i][adj[i]] if scheme == "bb"
                       else ge[io][adj[i]] + g[i][adj[i]] - ge[i][adj[i]])
            g[i][adj[i]] = 0.0
        for i, (cx, cy) in enumerate(VEL):
            if cx or cy:
                g[i] = np.roll(np.roll(g[i], cy, 0), cx, 1)
        for i in adj:
            tmp = g[OPP[i]].copy()
            tmp[adj[i]] += rets[i]
            g[OPP[i]] = tmp
        g[:, wall] = 0.0
        if not np.isfinite(g.sum()):
            return {"diverged_at_step": t + 1, "mass_over_initial": None}
    C = g.sum(axis=0)
    ref = C[channel & (dist > 4)].mean() if init == "uniform" else None
    out = {"mass_over_initial": float(g.sum()/m0), "diverged_at_step": None}
    if init == "uniform":
        out["wall_amp"] = float(np.abs(C[channel & (dist < 1)] - ref).max())
        out["mid_amp"] = float(np.abs(C[channel & (dist > 5)] - ref).max())
        out["layer_bins"] = {
            f"{lo}-{hi}": float(np.abs(
                C[channel & (dist >= lo) & (dist < hi)] - ref).max())
            for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6))}
    else:
        out["c_min"] = float(C[channel].min())
    return out


def run_nt(tau, u0, steps, profile, init="uniform"):
    """Noble-Torczynski partially-saturated cells, u_w = 0 in the solid
    operator (the u_w = local variant diverges under plug flow -
    measured; the record keeps the fact)."""
    ux, uy = _velocity(u0, profile)
    g = _init(init, ux, uy, notsolid)
    B = eps*(tau - 0.5)/((1.0 - eps) + (tau - 0.5))
    m0 = g.sum()
    for t in range(steps):
        C = g.sum(axis=0)
        ge = _feq(C, ux, uy)
        g0 = _feq(C, 0.0*ux, 0.0*uy)
        om = np.empty_like(g)
        for i in range(5):
            om[i] = (g[OPP[i]] - g0[OPP[i]]) - (g[i] - g0[i])
        g += (1.0 - B)*(-(g - ge)/tau) + B*om
        for i, (cx, cy) in enumerate(VEL):
            if cx or cy:
                g[i] = np.roll(np.roll(g[i], cy, 0), cx, 1)
        if not np.isfinite(g.sum()):
            return {"diverged_at_step": t + 1, "mass_over_initial": None}
    C = g.sum(axis=0)
    out = {"mass_over_initial": float(g.sum()/m0), "diverged_at_step": None}
    if init == "uniform":
        ref = C[interior & (dist > 4)].mean()
        out["wall_amp"] = float(np.abs(C[interior & (dist < 1)] - ref).max())
        out["mid_amp"] = float(np.abs(C[interior & (dist > 5)] - ref).max())
    else:
        out["c_min"] = float(C[notsolid].min())
    return out


def main():
    tau, u0, steps = 0.5048, 0.08, 20000
    rec = {
        "question": "a conservative, stable, accurate no-flux scalar wall "
                    "for advection oblique to the lattice",
        "rig": {"geometry": "doubly periodic oblique channel, slope 1/2, "
                            "P=50 W=30 box 100x100, D2Q5(2/3)",
                "null_test": "uniform C with Poiseuille profile across the "
                             "channel is a steady state of the continuous "
                             "problem; any deviation is the wall scheme's",
                "tau": tau, "u0": u0, "steps": steps},
        "schemes": {},
    }

    print("null tests ...")
    rec["schemes"]["bounce_back"] = {
        "plug": run_link_wall("bb", tau, u0, steps, False),
        "profile": run_link_wall("bb", tau, u0, steps, True),
        "blob_plug": run_link_wall("bb", tau, u0, steps, False, "blob"),
        "note": "conservative; bounded in this 2-D rig with an O(1) wall "
                "layer; the SAME wall diverges in 3-D D3Q7 at the pin "
                "(results/bifurcation_g4_control.json)",
    }
    rec["schemes"]["non_equilibrium_reflection"] = {
        "plug_uniform": run_link_wall("ner", tau, u0, steps, False),
        "profile": run_link_wall("ner", tau, u0, 6000, True),
        "blob_plug": run_link_wall("ner", tau, u0, 6000, False, "blob"),
        "outcome": "REFUTED: exact on uniform C + uniform u (machine-zero "
                   "deviation), but its per-link exchange is proportional "
                   "to local C - a positive-feedback pump that diverges "
                   "with any gradient",
    }
    rec["schemes"]["noble_torczynski"] = {
        "plug": run_nt(tau, u0, steps, False),
        "profile": run_nt(tau, u0, steps, True),
        "blob_plug": run_nt(tau, u0, steps, False, "blob"),
        "outcome": "the best conservative candidate: exact mass, stable "
                   "(u_w = 0 form), true-wall geometry, wall layer ~20% "
                   "below bounce-back's; the u_w = local form diverges "
                   "under plug flow",
    }
    rec["schemes"]["bouzidi_reflection"] = {
        "outcome": "REFUTED on the G4 control (3-D): stable axis-aligned "
                   "but retains 0.25 of the scalar - the interpolation is "
                   "not conservative and a scalar has no pressure field "
                   "to self-correct (results/bifurcation_g4_control.json)",
    }

    print("layer scaling ...")
    sweep = {"u_sweep_tau_0.5048": {}, "tau_sweep_u_0.08": {}}
    for uu in (0.02, 0.04, 0.08):
        r = run_link_wall("bb", tau, uu, 15000, True)
        sweep["u_sweep_tau_0.5048"][str(uu)] = {
            "wall_amp": r["wall_amp"], "amp_over_u": r["wall_amp"]/uu}
    for tt in (0.5048, 0.51, 0.53):
        r = run_link_wall("bb", tt, u0, 15000, True)
        sweep["tau_sweep_u_0.08"][str(tt)] = {"wall_amp": r["wall_amp"]}
    sweep["reading"] = ("amp ~ 5 u_lat, decreasing in (tau-1/2) with "
                        "measured exponent ~ -0.7; at the stability pin "
                        "refinement at fixed u_lat grows tau-1/2 "
                        "linearly, so the layer shrinks ~ res^0.7 - a "
                        "~1% window bias needs ~48 cells per radius")
    rec["layer_scaling"] = sweep

    print("Lambda sweep ...")
    lam_sweep = {}
    for lam, tag in (((tau-0.5)**2, "bgk_equiv"), (1/12, "advection_magic"),
                     (3/16, "wall_magic"), (1/4, "stability_magic")):
        r = run_link_wall("bb", tau, u0, 15000, True, lam=lam)
        lam_sweep[tag] = {"lambda": lam, "wall_amp": r["wall_amp"],
                          "mid_amp": r["mid_amp"]}
    lam_sweep["reading"] = ("no Lambda cures the layer (wall amp GROWS "
                            "with Lambda); Lambda = 1/12 minimises the "
                            "mid-channel spill, 4x below BGK")
    rec["trt_lambda_sweep"] = lam_sweep

    rec["paths_forward"] = [
        "(a) Noble-Torczynski walls + TRT Lambda=1/12 + high resolution "
        "(~48 cells/radius) on HPC - layer budgeted by the measured "
        "scaling; needs the NT ADE operator ported to OpenLB",
        "(b) a boundary-layer-corrected wall scheme - open theory: the "
        "layer is the equilibrium face-flux inconsistency, no collision "
        "knob removes it, and every conservative local correction "
        "measured so far either keeps it (BB, NT) or destabilises (NER)",
    ]
    rec["git_sha"] = git_sha(REPO)
    rec["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = REPO / "results" / "oblique_wall_scheme_study.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
