"""Error norms. Inputs are plain arrays already non-dimensionalised by u_ref."""

import numpy as np


def l2_velocity(numerical, analytic):
    """Discrete L2 norm of the velocity error at the sample points.

    Both inputs must be non-dimensional (u/u_max), evaluated at the same
    wall-normal stations, so the result is already normalised by u_max:

        E = sqrt( mean( (u_num/u_max - u_exact/u_max)**2 ) )
    """
    num = np.asarray(numerical, dtype=float)
    ana = np.asarray(analytic, dtype=float)
    if num.shape != ana.shape:
        raise ValueError(f"shape mismatch: numerical {num.shape} vs analytic {ana.shape}")
    return float(np.sqrt(np.mean((num - ana) ** 2)))


def l2_phase(numerical, analytic, weights):
    """Amplitude-weighted RMS phase error [radians] at the sample points.

    Phase differences are wrapped to (-pi, pi] before averaging, and weighted
    (typically by the squared analytic amplitude) because phase is
    ill-conditioned where the amplitude vanishes.
    """
    num = np.asarray(numerical, dtype=float)
    ana = np.asarray(analytic, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not (num.shape == ana.shape == w.shape):
        raise ValueError("shape mismatch between phases and weights")
    diff = np.angle(np.exp(1j * (num - ana)))
    return float(np.sqrt(np.sum(w * diff**2) / np.sum(w)))


def relative_error_scalar(numerical, analytic):
    """|numerical - analytic| / |analytic| for a scalar quantity.

    Both values must be in the same units; the result is dimensionless.
    """
    ana = float(analytic)
    if ana == 0.0:
        raise ValueError("analytic reference is zero; relative error undefined")
    return abs(float(numerical) - ana) / abs(ana)
