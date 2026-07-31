"""Standalone conservation check for an OpenFOAM case. READ-ONLY.

Deliberately NOT coupled to betaflow, to AortaCFD, or to any pipeline: it
takes a case directory, reads the written fields, and reports whether the
discrete conservation identity closes. It runs no solver and writes nothing
into the case, so it is safe to point at production output.

WHAT IT CHECKS. For an incompressible flow the face-flux field `phi` must sum
to zero over the whole boundary at every written time:

    sum over all boundary faces of phi = 0

phi is the volumetric flux through each face, positive along the face normal,
which points OUT of the domain on boundary patches. Any imbalance is fluid
appearing or vanishing. This is the same class of check as the momentum
identity used throughout betaflow: an exact relation the discrete solution
must satisfy, independent of the physics being modelled, so it measures
distance from a correct solution rather than iteration-to-iteration change.

WHY IT MATTERS FOR MULTI-OUTLET CASES. A mis-coupled outlet boundary condition
— a Windkessel with the wrong sign, a flow split that does not sum to the
inlet, an outlet left on a default BC — produces a plausible-looking velocity
and pressure field and a plausible-looking WSS map. Nothing in those outputs
reveals the error. The flux balance does, immediately.

FORMAT. Handles ascii and binary OpenFOAM fields, uniform and nonuniform
boundary entries. Binary lists are raw little-endian doubles between the
parentheses following the element count.
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

_PATCH_RE = re.compile(rb"\n {4}([A-Za-z_][\w.]*)\n {4}\{")


def _read_scalar_list(raw, start, end, binary):
    """Read a `nonuniform List<scalar> N ( ... )` within [start, end).

    The end bound matters: without it a patch whose entry is `uniform` (a wall,
    where the flux is identically zero) silently picks up the NEXT patch's
    list, which double-counts one boundary and invents an imbalance.
    """
    m = re.compile(rb"nonuniform\s+List<scalar>\s*(\d+)\s*\(").search(raw, start, end)
    if m is None:
        return None
    n = int(m.group(1))
    body = m.end()
    if binary:
        data = raw[body : body + 8 * n]
        if len(data) < 8 * n:
            return None
        return list(struct.unpack(f"<{n}d", data))
    close = raw.index(b")", body)
    return [float(x) for x in raw[body:close].split()]


def _patch_blocks(raw):
    """Yield (name, start, end) for each patch entry in boundaryField."""
    i = raw.find(b"boundaryField")
    if i < 0:
        return
    matches = list(_PATCH_RE.finditer(raw, i))
    for k, m in enumerate(matches):
        end = matches[k + 1].start() if k + 1 < len(matches) else len(raw)
        yield m.group(1).decode(), m.end(), end


def patch_fluxes(phi_path):
    """{patch: (net flux, n faces)} from a written phi field."""
    raw = phi_path.read_bytes()
    fmt = re.search(rb"format\s+(\w+)", raw)
    binary = bool(fmt and fmt.group(1) == b"binary")
    out = {}
    for name, start, end in _patch_blocks(raw):
        seg_end = end
        uniform = re.compile(rb"uniform\s+([-\d.eE+]+)\s*;").search(raw, start, seg_end)
        values = _read_scalar_list(raw, start, seg_end, binary)
        if values is not None:
            out[name] = (sum(values), len(values))
        elif uniform is not None:
            out[name] = (0.0, 0)  # uniform value with unknown face count
        else:
            out[name] = (0.0, 0)
    return out


def check_case(case_dir):
    """Flux closure at every written time of a case."""
    case_dir = Path(case_dir)
    times = []
    for d in case_dir.iterdir():
        if not d.is_dir():
            continue
        try:
            value = float(d.name)
        except ValueError:
            continue
        if (d / "phi").exists():
            times.append((value, d))
    times.sort()
    report = {"case": str(case_dir), "times": []}
    for value, d in times:
        fluxes = patch_fluxes(d / "phi")
        total = sum(f for f, _ in fluxes.values())
        scale = sum(abs(f) for f, _ in fluxes.values())
        report["times"].append({
            "time": value,
            "patches": {k: {"flux": f, "faces": n} for k, (f, n) in fluxes.items()},
            "net_flux": total,
            "throughput": scale,
            # Imbalance relative to the total flux crossing the boundary. This
            # is the number to read: an absolute flux means nothing without
            # the scale it is compared against.
            "relative_imbalance": (abs(total) / scale) if scale > 0 else float("nan"),
        })
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case", nargs="+", help="OpenFOAM case directory (read-only)")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="relative-imbalance threshold (default 1e-6)")
    ap.add_argument("--json", type=Path, help="write the full report here")
    args = ap.parse_args(argv)

    reports, worst = [], 0.0
    for case in args.case:
        r = check_case(case)
        reports.append(r)
        if not r["times"]:
            print(f"{case}: no written phi field found")
            continue
        last = r["times"][-1]
        worst = max(worst, last["relative_imbalance"])
        status = "OK" if last["relative_imbalance"] < args.tol else "IMBALANCE"
        print(f"\n{case}")
        print(f"  t = {last['time']:g}, {len(last['patches'])} patches")
        for name, info in last["patches"].items():
            print(f"    {name:<28} {info['flux']:+.6e}  ({info['faces']} faces)")
        print(f"    {'NET':<28} {last['net_flux']:+.6e}")
        print(f"    throughput {last['throughput']:.6e}"
              f"   relative imbalance {last['relative_imbalance']:.3e}   {status}")
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2) + "\n")
    return 0 if worst < args.tol else 1


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Momentum tier
# ---------------------------------------------------------------------------
#
# WHAT THIS TESTS, stated precisely. Summed over cells with the solver's own
# discrete operators, the momentum equation telescopes to its boundary terms,
# so a momentum balance built that way is implied by the solve just as the
# mass balance is implied by the pressure equation converging. This tier is
# NOT independent of the solve and does not claim to be.
#
# Its value is different: WSS AS PUBLISHED does not come from the solver's
# momentum assembly. The wallShearStress / forces functionObjects reconstruct
# it from snGrad and the viscosity model, a SEPARATE code path. Comparing that
# reconstruction against the balance the solver enforced tests the
# post-processing chain — which is where the published number actually comes
# from, and which nothing else checks.
#
# Traps handled here:
#   1. Wall TRACTION is not wall SHEAR STRESS: the traction carries -p n,
#      which on a curved wall does not vanish and is often larger than the
#      viscous part. Both are computed and reported separately.
#   2. Pressure reference: with all-Dirichlet velocity BCs p is defined up to
#      a constant, and the total is invariant only because the closed surface
#      integral of n vanishes. Omitting ANY patch makes the result
#      reference-dependent, so every patch is always included.
#   3. Closed-surface check first, as a pure mesh test.
#   4. Transient cases: cycle-average over a period rather than
#      finite-differencing volume-integrated momentum.

import numpy as np

from tools.foam_mesh import (  # noqa: E402
    face_area_vectors,
    read_boundary,
    read_faces,
    read_internal_field,
    read_labels,
    read_patch_field,
    read_points,
)


def mesh_boundary(case_dir):
    """Boundary face geometry, grouped by patch. Read-only."""
    pm = Path(case_dir) / "constant" / "polyMesh"
    boundary = read_boundary(pm / "boundary")
    points = read_points(pm / "points")
    faces = read_faces(pm / "faces")
    out = {}
    for name, info in boundary.items():
        s, n = info["startFace"], info["nFaces"]
        if n == 0:
            continue
        sf, cf = face_area_vectors(points, faces[s : s + n])
        out[name] = {"Sf": sf, "Cf": cf, "type": info["type"], "nFaces": n}
    return out


def closed_surface_residual(patches):
    """sum of Sf over EVERY boundary face; zero to round-off for a closed mesh.

    Pure geometry, and the first thing to run: it catches a missing or
    duplicated patch immediately, which is exactly the failure the mass-tier
    parser had.
    """
    total = np.zeros(3)
    area = 0.0
    for p in patches.values():
        total += p["Sf"].sum(axis=0)
        area += np.linalg.norm(p["Sf"], axis=1).sum()
    return total, area


def domain_volume(patches):
    """V = (1/3) closed-integral x . n dS — divergence theorem on the boundary.

    Needs no cell data, so the volume is available from the same read as the
    surface integrals.
    """
    return sum(float((p["Cf"] * p["Sf"]).sum()) for p in patches.values()) / 3.0


def _fill_from_owner(case_dir, time_dir, field, n_components, values, patches):
    """Reconstruct MISSING boundary values from the adjacent cell.

    A zeroGradient (or plain `calculated`) patch stores no value list, so its
    boundary values are simply absent from the written field. Omitting those
    patches from a traction integral is NOT a small error: with all-Dirichlet
    velocity BCs the pressure is defined only up to a constant, and the total
    is invariant only because the closed surface integral of n vanishes. Drop
    one patch and the answer becomes reference-dependent — it changes if
    someone alters pRefValue, which has nothing to do with the physics.

    zeroGradient means the face value EQUALS the owner cell value, so the
    owner list plus the internal field recovers it exactly.
    """
    missing = [k for k in patches if k not in values and patches[k]["nFaces"] > 0]
    if not missing:
        return values, []
    src = Path(time_dir) / field
    if not src.exists():
        return values, missing
    internal = read_internal_field(src, n_components)
    if internal is None or np.ndim(internal) == 0:
        return values, missing
    owner = read_labels(Path(case_dir) / "constant" / "polyMesh" / "owner")
    boundary = read_boundary(Path(case_dir) / "constant" / "polyMesh" / "boundary")
    filled = []
    for name in missing:
        info = boundary.get(name)
        if info is None:
            continue
        s, n = info["startFace"], info["nFaces"]
        cells = owner[s : s + n]
        if cells.max() >= len(internal):
            continue
        values[name] = internal[cells]
        filled.append(name)
    return values, [m for m in missing if m not in filled]


def momentum_terms(case_dir, time_dir, patches):
    """Per-patch pressure force, momentum flux and (walls) viscous force.

    Kinematic units throughout, matching incompressible OpenFOAM: p is p/rho
    and wallShearStress is tau/rho, so every term below is a force per unit
    density [m^4/s^2].
    """
    time_dir = Path(time_dir)
    nfaces = {k: v["nFaces"] for k, v in patches.items()}
    p_vals = read_patch_field(time_dir / "p", 1, nfaces) if (time_dir / "p").exists() else {}
    u_vals = read_patch_field(time_dir / "U", 3, nfaces) if (time_dir / "U").exists() else {}
    phi_vals = (
        read_patch_field(time_dir / "phi", 1, nfaces)
        if (time_dir / "phi").exists()
        else {}
    )
    wss_path = time_dir / "wallShearStress"
    wss_vals = read_patch_field(wss_path, 3, nfaces) if wss_path.exists() else {}
    # Fill zeroGradient/calculated patches from the owner cell — see the note
    # in _fill_from_owner on why omitting them is not a small error.
    p_vals, p_missing = _fill_from_owner(case_dir, time_dir, "p", 1, p_vals, patches)
    u_vals, u_missing = _fill_from_owner(case_dir, time_dir, "U", 3, u_vals, patches)

    # Area-weighted mean boundary pressure — the reference to remove.
    num = den = 0.0
    for name, geom in patches.items():
        if name in p_vals and len(p_vals[name]) == len(geom["Sf"]):
            area = np.linalg.norm(geom["Sf"], axis=1)
            num += float((p_vals[name] * area).sum())
            den += float(area.sum())
    p_mean = num / den if den > 0 else 0.0

    terms = {}
    for name, geom in patches.items():
        sf = geom["Sf"]
        entry = {"type": geom["type"], "nFaces": geom["nFaces"]}
        if name in p_vals and len(p_vals[name]) == len(sf):
            entry["pressure_force"] = (-p_vals[name][:, None] * sf).sum(axis=0)
            entry["pressure_force_demeaned"] = (
                -(p_vals[name] - p_mean)[:, None] * sf
            ).sum(axis=0)
        if name in u_vals and name in phi_vals and len(phi_vals[name]) == len(sf):
            entry["momentum_flux"] = (u_vals[name] * phi_vals[name][:, None]).sum(axis=0)
        if name in wss_vals and len(wss_vals[name]) == len(sf):
            mag = np.linalg.norm(sf, axis=1)
            entry["viscous_force"] = (wss_vals[name] * mag[:, None]).sum(axis=0)
        terms[name] = entry
    terms["_missing"] = {"p": p_missing, "U": u_missing, "p_mean": p_mean}
    return terms


def body_force_note(case_dir):
    """Flag a volume momentum source, which the boundary balance cannot see.

    A meanVelocityForce or semiImplicitSource adds momentum in the interior,
    so the boundary integrals alone will NOT close — by design, not by error.
    Reported rather than silently ignored.
    """
    case_dir = Path(case_dir)
    for rel in ("constant/fvModels", "system/fvConstraints", "constant/fvOptions"):
        f = case_dir / rel
        if f.exists():
            txt = f.read_text(errors="ignore")
            m = re.search(r"type\s+(\w*(?:VelocityForce|Source|acceleration))\s*;", txt)
            if m:
                return {"present": True, "file": rel, "type": m.group(1)}
    return {"present": False}


def momentum_closure(case_dir, time_dir=None, body_force=None):
    """Per-COMPONENT momentum closure over the whole boundary.

    Reports the three components separately, not a norm: a single component
    failing localises the error in a way a magnitude cannot.

    `body_force` is the uniform volume source per unit density [m/s^2], if the
    case has one (meanVelocityForce and friends). It is added as G*V, since
    a volume source does not appear in the boundary integrals at all.
    """
    case_dir = Path(case_dir)
    patches = mesh_boundary(case_dir)
    net_area, total_area = closed_surface_residual(patches)
    volume = domain_volume(patches)
    if time_dir is None:
        times = sorted(
            (d for d in case_dir.iterdir()
             if d.is_dir() and _is_time_name(d.name) and float(d.name) > 0),
            key=lambda d: float(d.name),
        )
        time_dir = times[-1]
    terms = momentum_terms(case_dir, time_dir, patches)

    total = np.zeros(3)
    parts = {"pressure_force": np.zeros(3), "momentum_flux": np.zeros(3),
             "viscous_force": np.zeros(3)}
    missing = terms.pop("_missing", {"p": [], "U": [], "p_mean": 0.0})
    for entry in terms.values():
        for key in parts:
            if key in entry:
                parts[key] += entry[key]
    # Momentum flux leaves the domain, so it sits on the other side of the
    # balance from the surface forces.
    total = parts["pressure_force"] + parts["viscous_force"] - parts["momentum_flux"]
    source = np.zeros(3)
    if body_force is not None:
        source = np.asarray(body_force, dtype=float) * volume
        total = total + source

    # SCALE IS THE GROSS MAGNITUDE, not the net. Normalising by the net is
    # meaningless whenever the terms cancel by construction — a Couette
    # channel's two walls sum to round-off, and dividing by that reports a
    # relative residual of 1.0 for a perfect balance. Summing the ABSOLUTE
    # per-patch contributions gives the throughput the residual should be
    # judged against, exactly as in the mass tier.
    gross = np.zeros(3)
    for entry in terms.values():
        for key in ("pressure_force", "viscous_force", "momentum_flux"):
            if key in entry:
                gross += np.abs(entry[key])
    gross += np.abs(source)
    scale = float(np.max(gross)) if np.max(gross) > 0 else 1e-300

    # REFERENCE-FREE SCALE. The gross pressure magnitude sum|p|A shifts to
    # |p0| sum(A) under p -> p + p0, so ANY normalisation built from it is
    # pressure-reference-dependent even though the identity itself is not
    # (the NET is invariant because the closed surface integral of n
    # vanishes). Measured here: the coronary cases carry a ~89 mmHg offset
    # with a pressure RANGE of only 1% of it, while BPM120 sits at ~37 mmHg
    # with a range comparable to its mean. Comparing cases on the raw gross
    # therefore compares their pressure references, not their physics.
    #
    # Subtracting the area-weighted mean pressure leaves the NET identity
    # untouched (same closed-surface argument) while making the gross
    # meaningful. That is the scale used for cross-case comparison.
    gross_free = np.zeros(3)
    for entry in terms.values():
        for key in ("viscous_force", "momentum_flux"):
            if key in entry:
                gross_free += np.abs(entry[key])
        if "pressure_force_demeaned" in entry:
            gross_free += np.abs(entry["pressure_force_demeaned"])
    gross_free += np.abs(source)
    scale_free = float(np.max(gross_free)) if np.max(gross_free) > 0 else 1e-300
    return {
        "time": float(Path(time_dir).name),
        "closed_surface_residual": net_area.tolist(),
        "closed_surface_relative": float(np.linalg.norm(net_area) / total_area),
        "domain_volume": volume,
        "terms": {k: v.tolist() for k, v in parts.items()},
        "body_force_term": source.tolist(),
        "missing_boundary_values": {k: v for k, v in missing.items() if k != "p_mean"},
        "mean_boundary_pressure": missing.get("p_mean", 0.0),
        "residual": total.tolist(),
        "gross_per_component": gross.tolist(),
        "gross_reference_free_per_component": gross_free.tolist(),
        "scale_reference_free": scale_free,
        "residual_relative_reference_free": (np.abs(total) / scale_free).tolist(),
        "residual_relative_per_component": (np.abs(total) / scale).tolist(),
        "scale": scale,
        "patches": {
            k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                for kk, vv in v.items()}
            for k, v in terms.items()
        },
    }


def _is_time_name(name):
    try:
        float(name)
        return True
    except ValueError:
        return False


def read_mean_velocity_force(case_dir):
    """The converged uniform source [m/s^2] applied by meanVelocityForce.

    Parsed from the solver log, which is the only record of what was actually
    applied — the dictionary states a target velocity, not the force.
    """
    for log in ("log.foamRun", "log.simpleFoam", "log.pimpleFoam"):
        f = Path(case_dir) / log
        if f.exists():
            hits = re.findall(r"pressure gradient = ([0-9eE.+-]+)", f.read_text())
            if hits:
                return float(hits[-1])
    return None
