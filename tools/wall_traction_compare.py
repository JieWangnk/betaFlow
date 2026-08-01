"""Wall traction, computed twice from the SAME solution. READ-ONLY.

The Stage A null test (a straight pipe with real inlet/outlet patches) leaves a
momentum residual of ~1e-3 relative where every cyclic case closes at 1e-16.
Three explanations were named and none was tested:

    (a) the checker's surface quadrature;
    (b) the wallShearStress functionObject's snGrad reconstruction differing
        from the solver's own momentum assembly -- which is what the momentum
        tier was BUILT to measure, and would be the production-relevant finding;
    (c) incomplete development leaving a real momentum flux.

This script separates them, and needs no new solve.

(a) is tested by geometry alone: the wedge's areas and volume have closed forms,
    so the quadrature can be differenced against them. The end faces matter
    here and have never been tested, because on a CYCLIC case the two ends
    cancel -- and so would any error in them.
(b) is tested by computing the wall viscous force twice: once by integrating
    the FO's written wallShearStress field, once from nu * snGrad(U) using the
    solver's own cell values and cell centres.
(c) is tested by x-invariance of the internal solution.

Nothing is written into the case directory.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from foam_mesh import (  # noqa: E402
    face_area_vectors,
    read_boundary,
    read_faces,
    read_internal_field,
    read_labels,
    read_patch_field,
    read_points,
)

WEDGE_HALF_ANGLE_DEG = 2.5


def cell_centres(points, faces, owner, neighbour, n_cells):
    """Cell centroids and volumes by OpenFOAM's pyramid decomposition.

    cEst = mean of the cell's face centres; each face then forms a pyramid of
    volume (1/3)(Cf - cEst).Sf_out with centroid (3/4)Cf + (1/4)cEst. Exact for
    a polyhedron with planar faces, which is what blockMesh produces here.
    """
    sf, cf = face_area_vectors(points, faces)
    acc_c = np.zeros((n_cells, 3))
    acc_n = np.zeros(n_cells)
    np.add.at(acc_c, owner, cf[: len(owner)])
    np.add.at(acc_n, owner, 1.0)
    np.add.at(acc_c, neighbour, cf[: len(neighbour)])
    np.add.at(acc_n, neighbour, 1.0)
    c_est = acc_c / acc_n[:, None]

    vol = np.zeros(n_cells)
    mom = np.zeros((n_cells, 3))
    for cells, sign, nf in ((owner, 1.0, len(owner)), (neighbour, -1.0, len(neighbour))):
        d = cf[:nf] - c_est[cells]
        pyr_v = sign * (d * sf[:nf]).sum(axis=1) / 3.0
        pyr_c = 0.75 * cf[:nf] + 0.25 * c_est[cells]
        np.add.at(vol, cells, pyr_v)
        np.add.at(mom, cells, pyr_v[:, None] * pyr_c)
    return mom / vol[:, None], vol, sf, cf


def read_nu(case_dir):
    txt = (Path(case_dir) / "constant" / "physicalProperties").read_text()
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("nu") and ";" in s:
            return float(s.rstrip(";").split()[-1])
    raise ValueError("no nu in constant/physicalProperties")


def latest_time(case_dir):
    times = []
    for d in Path(case_dir).iterdir():
        if d.is_dir():
            try:
                times.append((float(d.name), d))
            except ValueError:
                pass
    return max(times)[1] if times else None


def analyse(case_dir, radius=1.0, length=4.0, wall="wall"):
    case_dir = Path(case_dir)
    pm = case_dir / "constant" / "polyMesh"
    points, faces = read_points(pm / "points"), read_faces(pm / "faces")
    owner, neighbour = read_labels(pm / "owner"), read_labels(pm / "neighbour")
    boundary = read_boundary(pm / "boundary")
    n_cells = int(max(owner.max(), neighbour.max())) + 1
    cc, vol, sf, cf = cell_centres(points, faces, owner, neighbour, n_cells)

    th = np.radians(WEDGE_HALF_ANGLE_DEG)
    # Circumscribed wedge: vertices at a/cos(th), so the chord MIDPOINT sits on
    # r = a. Cross-section is a triangle of area a^2 tan(th); the wall face is
    # the chord, of half-width a tan(th).
    exact = {
        "cross_section": radius**2 * np.tan(th),
        "wall_area": 2.0 * radius * np.tan(th) * length,
        "volume": radius**2 * np.tan(th) * length,
    }

    nfaces = {k: v["nFaces"] for k, v in boundary.items() if v["nFaces"]}
    geom = {}
    for name, info in boundary.items():
        if not info["nFaces"]:
            continue
        s, n = info["startFace"], info["nFaces"]
        geom[name] = {"Sf": sf[s : s + n], "Cf": cf[s : s + n],
                      "start": s, "n": n, "type": info["type"]}

    # --- (a) quadrature against closed forms --------------------------------
    areas = {k: float(np.linalg.norm(v["Sf"], axis=1).sum()) for k, v in geom.items()}
    closed = sum(v["Sf"].sum(axis=0) for v in geom.values())
    vol_boundary = sum(float((v["Cf"] * v["Sf"]).sum()) for v in geom.values()) / 3.0
    quadrature = {
        "closed_surface_over_area": float(np.linalg.norm(closed) / sum(areas.values())),
        "volume_from_boundary": vol_boundary,
        "volume_from_cells": float(vol.sum()),
        "volume_exact": exact["volume"],
        "volume_rel_error": abs(vol_boundary / exact["volume"] - 1.0),
        "wall_area_rel_error": abs(areas[wall] / exact["wall_area"] - 1.0),
    }
    for end in ("inlet", "outlet"):
        if end in areas:
            quadrature[f"{end}_area_rel_error"] = abs(
                areas[end] / exact["cross_section"] - 1.0)

    # --- (b) FO versus assembly --------------------------------------------
    t = latest_time(case_dir)
    nu = read_nu(case_dir)
    u_int = read_internal_field(t / "U", 3)
    wss = read_patch_field(t / "wallShearStress", 3, nfaces)
    u_pat = read_patch_field(t / "U", 3, nfaces)

    g = geom[wall]
    mag = np.linalg.norm(g["Sf"], axis=1)
    n_hat = g["Sf"] / mag[:, None]
    fo_force = (wss[wall] * mag[:, None]).sum(axis=0)

    own = owner[g["start"] : g["start"] + g["n"]]
    u_c = u_int[own]
    u_f = u_pat[wall] if wall in u_pat else np.zeros_like(u_c)
    d_n = ((g["Cf"] - cc[own]) * n_hat).sum(axis=1)          # wall-normal distance
    sn_grad = (u_f - u_c) / d_n[:, None]                     # OpenFOAM's snGrad
    sn_grad -= (sn_grad * n_hat).sum(axis=1)[:, None] * n_hat  # tangential part
    asm_force = nu * (sn_grad * mag[:, None]).sum(axis=0)

    # --- (c) development ----------------------------------------------------
    # Group by streamwise station and compare radial profiles between the
    # first and last. NOTE: cell-centre x carries floating-point jitter of
    # order 1e-16, so grouping must round before it sorts. An earlier version
    # lexsorted on the raw x and reported an 85% profile change -- it was
    # comparing r = 0.16 against r = 0.94, and it "confirmed" this hypothesis
    # immediately after the other two had been refuted.
    x_key = np.round(cc[:, 0], 9)
    xs = np.unique(x_key)
    rad = np.hypot(cc[:, 1], cc[:, 2])

    def profile(xi):
        m = np.flatnonzero(x_key == xi)
        return m[np.argsort(rad[m])]

    p_in, p_out = profile(xs[0]), profile(xs[-1])
    u_in, u_out = u_int[p_in, 0], u_int[p_out, 0]
    # Momentum flux through a station, integrated on the station's own cells:
    # sum u^2 dA, with dA the cell's share of the cross-section.
    def flux(idx):
        w = vol[idx] / (cc[idx, 0].max() - cc[idx, 0].min() + length / len(xs))
        return float((u_int[idx, 0] ** 2 * w).sum())
    development = {
        "n_x_stations": int(len(xs)),
        "max_rel_profile_change": float(
            np.max(np.abs(u_out - u_in)) / np.max(np.abs(u_in))),
        "u_max_first_station": float(u_in.max()),
        "u_max_last_station": float(u_out.max()),
        "momentum_flux_rel_change": float(abs(flux(p_out) / flux(p_in) - 1.0)),
    }

    return {
        "case": case_dir.name,
        "time": t.name,
        "n_cells": n_cells,
        "nu": nu,
        "quadrature": quadrature,
        "wall_force_fo_x": float(fo_force[0]),
        "wall_force_assembly_x": float(asm_force[0]),
        "fo_minus_assembly_over_fo": float(
            (fo_force[0] - asm_force[0]) / fo_force[0]),
        "development": development,
    }


def interior_balance(case_dir, length=4.0, wall="wall", split=0.5):
    """Momentum balance over a control volume that EXCLUDES the inlet patch.

    Its upstream face is an interior x-plane, where the face value is the
    solver's own central interpolation and the flux is the solver's own phi.
    If the residual is a boundary-treatment error at the inlet, this closes
    better and faster than the full domain; if it is distributed, it does not.
    """
    case_dir = Path(case_dir)
    pm = case_dir / "constant" / "polyMesh"
    points, faces = read_points(pm / "points"), read_faces(pm / "faces")
    owner, neighbour = read_labels(pm / "owner"), read_labels(pm / "neighbour")
    boundary = read_boundary(pm / "boundary")
    n_cells = int(max(owner.max(), neighbour.max())) + 1
    cc, vol, sf, cf = cell_centres(points, faces, owner, neighbour, n_cells)
    t = latest_time(case_dir)
    p_int = read_internal_field(t / "p", 1)
    u_int = read_internal_field(t / "U", 3)
    phi_int = read_internal_field(t / "phi", 1)
    nfaces = {k: v["nFaces"] for k, v in boundary.items() if v["nFaces"]}
    wss = read_patch_field(t / "wallShearStress", 3, nfaces)
    phi_p = read_patch_field(t / "phi", 1, nfaces)

    def patch(nm):
        i = boundary[nm]
        s, k = i["startFace"], i["nFaces"]
        return sf[s : s + k], cf[s : s + k], owner[s : s + k]

    _, _, own_out = patch("outlet")
    sf_w, cf_w, _ = patch(wall)
    x_cut = split * length
    pl = np.flatnonzero(np.isclose(cf[: len(phi_int), 0], x_cut, atol=1e-9))
    o_pl, n_pl = owner[pl], neighbour[pl]
    p_face = 0.5 * (p_int[o_pl] + p_int[n_pl])
    u_face = 0.5 * (u_int[o_pl] + u_int[n_pl])
    # sf on an interior face points owner -> neighbour (+x); the CV's outward
    # normal on its upstream face is -that, so -p*n = +p*Sf.
    f_p = float((p_face[:, None] * sf[pl]).sum(axis=0)[0])
    f_m = float(-(u_face[:, 0] * phi_int[pl]).sum()
                + (u_int[own_out, 0] * phi_p["outlet"]).sum())
    keep = cf_w[:, 0] > x_cut
    f_v = float((wss[wall][keep]
                 * np.linalg.norm(sf_w[keep], axis=1)[:, None]).sum(axis=0)[0])
    res = f_p + f_v - f_m
    return {"pressure_x": f_p, "viscous_x": f_v, "flux_x": f_m,
            "residual_x": res,
            "relative": abs(res) / (abs(f_p) + abs(f_v) + abs(f_m))}


if __name__ == "__main__":
    import json
    for d in sys.argv[1:]:
        out = analyse(d)
        out["interior_control_volume"] = interior_balance(d)
        print(json.dumps(out, indent=1))
