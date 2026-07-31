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
