"""Analytic reference for pulsatile (Womersley-type) flow in a plane channel.

Geometry: plane channel, half-height h, no-slip walls at y = +/- h, driven by
a spatially uniform oscillatory KINEMATIC pressure gradient

    G(t) = G cos(omega t)        [m/s^2]

Writing u(y, t) = Re{ uhat(y) e^{i omega t} }, the exact solution is

    uhat(y) = (G / (i omega)) * (1 - cosh(k y) / cosh(k h))
    k       = (1 + i) / delta,   delta = sqrt(2 nu / omega)  (Stokes layer)

WOMERSLEY NUMBER DEFINITION:

    alpha = h * sqrt(omega / nu)

    * length scale: the HALF-height h (walls at +/- h)
    * so delta / h = sqrt(2) / alpha, and k h = (1 + i) alpha / sqrt(2)

NOTE ON BESSEL FUNCTIONS: the classical Womersley solution built on J0 Bessel
functions of complex argument is the CIRCULAR PIPE solution. For the plane
channel the kernel is the complex cosh above — same physics, same
alpha-scaling, different geometry. Using J0 against a plane-channel solver
would be a silent geometry mismatch of exactly the kind this harness exists
to catch.

Scales: all velocities here are normalised by u_ref = G / omega (the
inviscid-core oscillation amplitude), and wall shear by tau_ref = G * h (the
quasi-steady wall shear). In those units:

    uhat / u_ref          = -i (1 - cosh(K y/h) / cosh(K)),  K = (1+i) alpha / sqrt(2)
    <uhat> / u_ref        = -i (1 - tanh(K) / K)
    tauhat / tau_ref      = tanh(K) / K       (kinematic, either wall)

The momentum-balance identity holds exactly:

    tau_w(t) = h * (G(t) - d<u>/dt)

(verified symbolically: h(G - i omega <uhat>) = G h tanh(K)/K = tauhat), and
as alpha -> 0 the flow reduces to quasi-steady Poiseuille (tanh(K)/K -> 1).
The flow is LINEAR (convection vanishes identically for u = u(y) x_hat), so
no Reynolds number enters — amplitude is a free scale.
"""

import numpy as np


def stokes_layer(nu, omega):
    """Stokes layer thickness delta = sqrt(2 nu / omega) [m]."""
    return np.sqrt(2.0 * nu / omega)


def womersley_number(h, omega, nu):
    """alpha = h sqrt(omega/nu) — THE definition used here (half-height h)."""
    return h * np.sqrt(omega / nu)


def _K(alpha):
    return (1.0 + 1.0j) * alpha / np.sqrt(2.0)


def complex_profile(y_over_h, alpha):
    """Complex velocity profile uhat / u_ref at stations y/h in [-1, 1].

    u(y, t) = u_ref * Re{ complex_profile * e^{i omega t} }, u_ref = G/omega.
    Amplitude is abs(), phase (radians, relative to the G cos(omega t)
    forcing) is angle().
    """
    y = np.asarray(y_over_h, dtype=float)
    K = _K(alpha)
    return -1.0j * (1.0 - np.cosh(K * y) / np.cosh(K))


def amplitude(y_over_h, alpha):
    """|uhat| / u_ref at stations y/h."""
    return np.abs(complex_profile(y_over_h, alpha))


def phase(y_over_h, alpha):
    """arg(uhat) [radians] at stations y/h, relative to the forcing cosine."""
    return np.angle(complex_profile(y_over_h, alpha))


def complex_bulk(alpha):
    """Complex bulk velocity <uhat> / u_ref."""
    K = _K(alpha)
    return -1.0j * (1.0 - np.tanh(K) / K)


def complex_wall_shear(alpha):
    """Complex kinematic wall shear tauhat / tau_ref (tau_ref = G h).

    Identical magnitude and phase at both walls. Equals tanh(K)/K, which is
    also h(G - d<u>/dt) in these units — the conservation identity.
    As alpha -> 0 this tends to 1 (quasi-steady Poiseuille tau = G h);
    for large alpha, |tanh(K)/K| -> 1/alpha exactly (|K| = alpha, tanh -> 1;
    measured 1.000000/alpha at alpha = 100) — the WSS becomes a small
    difference between the two large terms G and d<u>/dt. CORRECTED: this
    line said sqrt(2)/alpha, conflating |K| with alpha/sqrt(2); the code was
    always right and the pipe/channel ratio (exactly 2) is unaffected.
    """
    K = _K(alpha)
    return np.tanh(K) / K


def velocity(y_over_h, t_over_T, alpha):
    """u / u_ref at stations y/h and time t/T (T = 2 pi / omega)."""
    t = np.asarray(t_over_T, dtype=float)
    return np.real(
        np.outer(np.exp(2.0j * np.pi * t), complex_profile(y_over_h, alpha))
    ).squeeze()
