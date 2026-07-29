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
