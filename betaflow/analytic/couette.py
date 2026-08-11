"""Analytic reference for steady, laminar, plane Couette flow.

Geometry: gap of height H between parallel plates; the wall at y = 0 is
fixed, the wall at y = H moves at speed U in +x. No pressure gradient and no
body force — the motion is driven entirely by the moving wall.

    u(y)  = U * y / H       (pure shear, linear)
    tau_w = mu * U / H      (uniform shear: identical at both walls)

REYNOLDS NUMBER DEFINITION:

    Re = U * H / nu

    * velocity scale: the MOVING-WALL speed U
    * length scale:   the FULL gap height H

(The plane-Poiseuille case uses bulk velocity and full channel height 2h —
different flow, different convention; each analytic reference states its own. Never carry
one case's Re definition into another.)

Null-test role: the exact profile is linear. The second-order interior
scheme, the half-cell one-sided wall gradient, and linear (cellPoint)
interpolation are all EXACT for linear fields, so every numerical error in
this case should sit at round-off on every mesh — there is no discretisation
error to converge. A deviation is a harness or solver bug, not a mesh effect.
"""

import numpy as np


def velocity_profile(y_over_H):
    """Non-dimensional velocity u/U at stations y/H.

    Parameters
    ----------
    y_over_H : array_like
        Wall-normal coordinate normalised by the gap height, in [0, 1].
        0 is the fixed wall, 1 the moving wall.

    Returns
    -------
    ndarray
        u/U = y/H.
    """
    return np.asarray(y_over_H, dtype=float)


def tau_wall(mu, u_wall, H):
    """Wall shear stress mu * U / H — uniform, identical at both walls.

    Unit-agnostic: dynamic viscosity mu [Pa s] gives tau in Pa; kinematic
    viscosity nu [m^2/s] gives the kinematic tau [m^2/s^2] that
    incompressible OpenFOAM reports.
    """
    return mu * u_wall / H


def reynolds(u_wall, H, nu):
    """Re = u_wall * H / nu — THE definition for this case (see module docstring)."""
    return u_wall * H / nu
