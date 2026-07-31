"""Generate a 5-block O-grid blockMeshDict for a circular pipe.

STATUS: standalone, NOT yet wired into betaflow/runners/. It exists because
the O-grid is required for Lagrangian particles later (Brownian displacement
is isotropic and a wedge has no azimuthal room), and because re-running the
pipe cases on a second topology is the mesh-topology test.

WHAT IT ESTABLISHED. Running pipe Poiseuille on this mesh (8000 hex, 80
circumferential faces) gave a conservation-identity gap of 7.710e-04 against
a predicted 1 - cos(pi/80) = 7.7096e-04 — the SAME polygonal-faceting law
found on the wedge, where 1 - cos(2.5 deg) = 9.5178e-04 was measured as
9.518e-04. A polygon with its vertices ON the circle has
area/perimeter = R cos(pi/n)/2, so tau_w = G R cos(pi/n)/2 rather than G R/2.

The fix is the same in both topologies: scale the boundary radius by
1/cos(pi/n) so the face MIDPOINTS lie on the true circle. That correction is
applied in the wedge path (see _wedge_params in the OpenFOAM runner); it is
NOT applied here yet, which is why this script is a tool rather than a runner
template.

The two topologies differ in one important way: the wedge half-angle is fixed
by the mesh construction, so its bias never shrinks under refinement at all,
whereas the O-grid bias falls as 1/n_circ^2 — but only under CIRCUMFERENTIAL
refinement. A radial-only convergence study leaves both untouched.
"""
import numpy as np, pathlib, sys

def ogrid_dict(a, length, n_c, n_r, nx, f=0.5):
    import numpy as np, pathlib, sys

def ogrid_dict(a, length, n_c, n_r, nx, f=0.5):
    """5-block O-grid for a circular pipe. x is axial; cross-section in (y,z).

    Outer vertices sit ON the circle at 45 deg increments, joined by arc
    edges through the axis-aligned points so the boundary is a true circle,
    not a polygon — the O-grid's answer to the wedge faceting problem.
    """
    s = a / np.sqrt(2.0)
    inner = [(f*s, f*s), (-f*s, f*s), (-f*s, -f*s), (f*s, -f*s)]
    outer = [(s, s), (-s, s), (-s, -s), (s, -s)]
    pts = inner + outer
    v = []
    for x in (0.0, length):
        for (y, z) in pts:
            v.append((x, y, z))
    lines = ["FoamFile\n{\n    format ascii;\n    class dictionary;\n    object blockMeshDict;\n}\n",
             "convertToMeters 1;\n", "vertices\n("]
    for (x, y, z) in v:
        lines.append(f"    ({x:.12g} {y:.12g} {z:.12g})")
    lines.append(");\n")
    # blocks: central + 4 outer.  local index +8 gives the x=L plane
    lines.append("blocks\n(")
    # Cross-section vertices run counterclockwise in (y,z); traversing them
    # that way makes the hex left-handed (blockMesh: "inside-out"), so each
    # block lists its cross-section CLOCKWISE.
    lines.append(f"    hex (0 1 2 3 8 9 10 11) ({n_c} {n_c} {nx}) simpleGrading (1 1 1)")
    for k in range(4):
        i0, i1 = k, (k+1) % 4
        o0, o1 = 4+k, 4+(k+1) % 4
        lines.append(f"    hex ({i1} {i0} {o0} {o1} {i1+8} {i0+8} {o0+8} {o1+8}) "
                     f"({n_c} {n_r} {nx}) simpleGrading (1 1 1)")
    lines.append(");\n")
    # arc edges on both planes, midpoint on the circle between the 45deg pts
    lines.append("edges\n(")
    mids = [(0.0, a), (-a, 0.0), (0.0, -a), (a, 0.0)]
    for plane in (0, 8):
        for k in range(4):
            o0, o1 = 4+k+plane, 4+((k+1) % 4)+plane
            my, mz = mids[k]
            x = 0.0 if plane == 0 else length
            lines.append(f"    arc {o0} {o1} ({x:.12g} {my:.12g} {mz:.12g})")
    lines.append(");\n")
    lines.append("boundary\n(")
    lines.append("    inlet\n    {\n        type cyclic;\n        neighbourPatch outlet;\n        faces\n        (")
    lines.append("            (0 1 2 3)")
    for k in range(4):
        i0, i1 = k, (k+1) % 4
        lines.append(f"            ({i0} {i1} {4+(k+1)%4} {4+k})")
    lines.append("        );\n    }")
    lines.append("    outlet\n    {\n        type cyclic;\n        neighbourPatch inlet;\n        faces\n        (")
    lines.append("            (8 11 10 9)")
    for k in range(4):
        i0, i1 = k+8, (k+1) % 4 + 8
        lines.append(f"            ({i0} {4+k+8} {4+(k+1)%4+8} {i1})")
    lines.append("        );\n    }")
    lines.append("    wall\n    {\n        type wall;\n        faces\n        (")
    for k in range(4):
        o0, o1 = 4+k, 4+(k+1) % 4
        lines.append(f"            ({o0} {o1} {o1+8} {o0+8})")
    lines.append("        );\n    }")
    lines.append(");\n")
    return "\n".join(lines)

out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
(out/"blockMeshDict").write_text(ogrid_dict(1.0, 0.5, 20, 20, 4))
