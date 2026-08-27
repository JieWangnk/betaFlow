#!/usr/bin/env python3
"""Path (b) of the oblique-wall programme: closure theory, first rung.

The oblique-wall layer (results/oblique_wall_scheme_study.json) needs a
theory before a corrected scheme can be designed rather than guessed.
This tool holds the first reduced model — the x-invariant D2Q5 half-space
with uniform drift and a wall closure at one end — and the three results
it produced, two of them exclusions with mechanisms and one a design
clue. The model's own limit is the fourth result.

1. THE SINGLE-POINT CLOSURE FAMILY IS EXHAUSTED. Closures
   g_+ = g_-* + alpha * 2 w v C(0) / cs2 interpolate bounce-back
   (alpha = 0) to the non-equilibrium reflection wall (alpha = 1). Under
   drift INTO the wall the spectral radius grows monotonically with
   alpha; bounce-back sits exactly at the marginal point (max|lambda| =
   1.000000) and nothing above it is stable. The 2-D NER divergence is
   therefore not an implementation accident: no correction of this form
   can be stable.

2. THE WALL MODE HAS CHECKERBOARD PARITY — the design clue. Sourcing
   the same correction from the concentration at neighbour j = 1 or
   j = 3 (odd sites) is spectrally STABLE at every alpha tested
   (max|lambda| ~ 0.996), while j = 0 and j = 2 (even sites) are
   unstable: the growing mode alternates sign site-to-site, and odd
   sourcing feeds it in antiphase. The trade measured with it: the
   steady state under odd sourcing carries an O(1) checkerboard layer —
   stability was bought, accuracy was not.

3. DRIFT SIGN ASYMMETRY: with drift OUT of the wall every closure
   tested is stable. The instability lives only on the into-flow faces,
   matching the 2-D staircase picture where face orientations alternate.

4. THE MODEL'S LIMIT, stated so the next rung aims correctly: an
   x-invariant half-space with uniform normal drift carries NET flux
   into the wall — a physically different problem (its steady state is a
   genuine accumulation layer, flux balance -D dC/dy = v C). The
   oblique-channel artifact lives in the ALTERNATION of face normals
   along the staircase, which x-invariance cannot represent. The
   correctly posed object for the next rung is the staircase-periodic
   half-space: Bloch modes along the wall with the (2,1) staircase unit
   cell, a per-wavenumber transfer eigenproblem.

Run: python3 tools/halfspace_closure_study.py   (~1 min)
Writes: results/halfspace_closure_study.json
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from betaflow.provenance import git_sha  # noqa: E402

CS2 = 1.0/3.0
CY = np.array([0, 0, 0, 1, -1])
CX = np.array([0, 1, -1, 0, 0])
WGT = np.array([1.0/3.0] + [1.0/6.0]*4)


def _update_matrix(J, tau, ux, uy, alpha, j_src):
    """collide + stream + closure as a dense matrix; wall below site 0,
    closure correction alpha * 2 w (-uy)/cs2 * C(j_src)."""
    n = 5*J

    def idx(j, i):
        return 5*j + i

    q = WGT * (1.0 + (CX*ux + CY*uy)/CS2)
    Mc = np.zeros((n, n))
    for j in range(J):
        for i in range(5):
            Mc[idx(j, i), idx(j, i)] += 1.0 - 1.0/tau
            for k in range(5):
                Mc[idx(j, i), idx(j, k)] += q[i]/tau
    Ms = np.zeros((n, n))
    for j in range(J):
        for i in (0, 1, 2):
            Ms[idx(j, i), idx(j, i)] = 1.0
        if j >= 1:
            Ms[idx(j, 3), idx(j-1, 3)] = 1.0
        if j <= J-2:
            Ms[idx(j, 4), idx(j+1, 4)] = 1.0
    Ms[idx(0, 3), idx(0, 4)] = 1.0
    for k in range(5):
        Ms[idx(0, 3), idx(j_src, k)] += alpha * 2.0 * WGT[3] * (-uy)/CS2
    return Ms @ Mc


def _radius(J, tau, ux, uy, alpha, j_src):
    M = _update_matrix(J, tau, ux, uy, alpha, j_src)
    for i in range(5):
        M[5*(J-1)+i, :] = 0.0          # absorbing far field
    return float(np.abs(np.linalg.eigvals(M)).max())


def _steady_wall_values(J, tau, ux, uy, alpha, j_src):
    M = _update_matrix(J, tau, ux, uy, alpha, j_src)
    q = WGT * (1.0 + (CX*ux + CY*uy)/CS2)
    A = M - np.eye(5*J)
    b = np.zeros(5*J)
    for i in range(5):
        r = 5*(J-1) + i
        A[r, :] = 0.0
        A[r, r] = 1.0
        b[r] = q[i]
    g = np.linalg.lstsq(A, b, rcond=None)[0].reshape(J, 5)
    C = g.sum(axis=1)
    return [float(v) for v in (C[:6] - C[-1])]


def main():
    tau, u0 = 0.5048, 0.08
    th = np.arctan(0.5)
    ux, uy = u0*np.cos(th), -u0*np.sin(th)
    J = 150

    rec = {
        "question": "is there a stable wall closure with a smaller layer "
                    "than bounce-back's",
        "model": "x-invariant D2Q5 half-space, uniform drift, closure "
                 "family g+ = g-* + alpha 2 w v C(j_src)/cs2; "
                 f"tau={tau}, u0={u0}, direction slope 1/2, J={J}",
        "spectral_radius": {},
        "steady_wall_values": {},
    }

    for j_src in (0, 1, 2, 3):
        row = {}
        for alpha in (0.0, 0.25, 0.5, 1.0):
            row[str(alpha)] = _radius(J, tau, ux, uy, alpha, j_src)
        rec["spectral_radius"][f"into_wall_jsrc_{j_src}"] = row
    rec["spectral_radius"]["out_of_wall_local"] = {
        str(a): _radius(J, tau, ux, -uy, a, 0) for a in (0.0, 0.5, 1.0)}

    rec["steady_wall_values"]["bb"] = _steady_wall_values(
        J, tau, ux, uy, 0.0, 0)
    rec["steady_wall_values"]["odd_sourced_alpha1"] = _steady_wall_values(
        J, tau, ux, uy, 1.0, 1)

    rec["findings"] = [
        "single-point closure family exhausted: spectral radius grows "
        "monotonically with alpha under into-wall drift; bounce-back is "
        "exactly marginal, nothing above it is stable",
        "the wall mode has checkerboard parity: odd-neighbour sourcing "
        "(j_src = 1, 3) is stable at every alpha (~0.996) while local and "
        "even sourcing are unstable - a design clue for corrected schemes",
        "odd sourcing trades the instability for an O(1) checkerboard "
        "steady layer: stability bought, accuracy not",
        "drift OUT of the wall: every closure stable - the instability "
        "lives on into-flow faces only",
        "MODEL LIMIT: x-invariant drift into a no-flux wall is the "
        "physical accumulation problem, not the oblique-staircase "
        "artifact; the next rung is the staircase-periodic half-space "
        "(Bloch modes along the wall, (2,1) unit cell)",
    ]
    rec["git_sha"] = git_sha(REPO)
    rec["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = REPO / "results" / "halfspace_closure_study.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
