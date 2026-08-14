"""Analytic reference for steady, laminar, plane Poiseuille flow.

Geometry: infinite plane channel of half-height h, stationary no-slip walls at
y = +/- h, constant streamwise pressure gradient G = -dp/dx > 0 (equivalently a
constant body force) driving flow in +x. Fully developed, so u = u(y) only.

    u(y)   = (G / (2 mu)) * (h**2 - y**2)
    u_max  = G * h**2 / (2 mu)        centreline velocity, at y = 0
    u_mean = (2/3) * u_max            bulk (area-averaged) velocity

REYNOLDS NUMBER DEFINITION — read this before comparing against anything:

    Re = u_mean * (2 h) / nu

    * velocity scale: the BULK (mean) velocity u_mean, NOT the centreline u_max
    * length scale:   the FULL channel height 2h,     NOT the half-height h

Mismatched Re definitions are the single most common source of silent
disagreement between "identical" channel-flow setups. Common alternatives and
their conversion to this definition:

    Re_c = u_max * h / nu      (centreline velocity, half-height)
         = (3/4) * Re                  [u_max = (3/2) u_mean, h = (2h)/2]
    Re_D = u_mean * D_h / nu   (hydraulic diameter D_h = 4h for a plane channel)
         = 2 * Re

Every function here is Re-independent because the non-dimensional profile
u/u_max vs y/h is a universal parabola; Re only enters when a runner converts
the case's Re into a viscosity. Use `reynolds()` to cross-check that a runner
used the same definition.
"""

import numpy as np


def velocity_profile(y_over_h):
    """Non-dimensional velocity u/u_max at wall-normal stations y/h.

    Parameters
    ----------
    y_over_h : array_like
        Wall-normal coordinate normalised by the half-height, in [-1, 1].
        Walls are at y/h = -1 and +1; the centreline is y/h = 0.

    Returns
    -------
    ndarray
        u/u_max = 1 - (y/h)**2.
    """
    y = np.asarray(y_over_h, dtype=float)
    return 1.0 - y**2


# Exact for plane Poiseuille: u_mean = (2/3) u_max.
U_MAX_OVER_U_MEAN = 1.5


def u_max(pressure_gradient, mu, h):
    """Centreline velocity G*h^2/(2*mu) for G = -dp/dx [Pa/m], mu [Pa s], h [m]."""
    return pressure_gradient * h**2 / (2.0 * mu)


def u_mean(pressure_gradient, mu, h):
    """Bulk velocity (2/3)*u_max."""
    return (2.0 / 3.0) * u_max(pressure_gradient, mu, h)


def reynolds(u_mean, h, nu):
    """Re = u_mean * (2 h) / nu — THE definition used throughout betaflow.

    Bulk velocity, full channel height. See the module docstring before
    substituting any other convention.
    """
    return u_mean * (2.0 * h) / nu


def pressure_gradient(u_mean, h, mu):
    """Pressure gradient G = -dp/dx sustaining bulk velocity u_mean: 3 mu u_mean / h**2.

    Any consistent units: dynamic viscosity mu [Pa s] gives G in Pa/m, kinematic
    viscosity nu [m^2/s] for the kinematic G [m/s^2] used by incompressible
    OpenFOAM. From u_mean = (2/3) u_max = G h**2 / (3 mu).
    """
    return 3.0 * mu * u_mean / h**2


def tau_wall(pressure_gradient, h):
    """Wall shear stress: tau_w = G * h, exact for plane Poiseuille.

    From tau = mu du/dy at y = -h with du/dy = G h / mu; equivalently the
    force balance on a channel slab, G * (2 h) = 2 tau_w. Any consistent units, like
    `pressure_gradient`: dynamic G [Pa/m] gives tau_w in Pa; kinematic G
    [m/s^2] gives the kinematic tau_w [m^2/s^2] that incompressible OpenFOAM
    reports.
    """
    return pressure_gradient * h
