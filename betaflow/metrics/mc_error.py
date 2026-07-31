"""Monte Carlo error laws — one per ESTIMATOR, with derivations.

There is no "the Monte Carlo error law". There is the law for THIS estimator
on THIS fit, and the four below differ by up to a factor of two. Using one
where another belongs is the statistical analogue of reusing a channel oracle
on a pipe: the number looks reasonable and is wrong by an O(1) factor.

Every law here is a relative standard deviation, to be divided by sqrt(N).

DERIVATION, common structure. For Brownian motion the squared displacement
has covariance proportional to min(s,t)^2:

    Cov(|r(s)|^2, |r(t)|^2) = 24 D^2 min(s,t)^2        (3-D)
    Cov(x(s)^2,   x(t)^2)   =  8 D^2 min(s,t)^2        (1-D)

Setting s = t recovers the single-time variances: 24 D^2 t^2 = (2Dt)^2 * 2*3
for chi^2_3, and 8 D^2 t^2 = (2Dt)^2 * 2 for chi^2_1.

A through-origin OLS slope over times uniform on (0, T] has

    Var(beta) = C * [ int int u v min(u,v)^2 du dv ] / [ int u^2 du ]^2
              = C * (1/12) / (1/9) = (3/4) C

where C is the single-time covariance coefficient. So for ANY process with
Cov proportional to min(s,t)^2:

    sd(slope fit) / sd(single final time) = sqrt(3/4) = 0.8660

which is pure fit geometry — the physical coefficient cancels. The fit beats
a single time because it uses correlated but not redundant information.
"""

import math

# Ratio of a through-origin slope fit to a single-time estimate, for any
# process with Cov proportional to min(s,t)^2. Pure fit geometry.
FIT_OVER_SINGLE_TIME = math.sqrt(3.0 / 4.0)  # 0.8660254

# 3-D mean squared displacement, |r|^2/(2Dt) ~ chi^2_3.
#   relative sd at one time = sqrt(2*3)/3 = sqrt(6)/3
SINGLE_TIME_MSD_3D = math.sqrt(6.0) / 3.0            # 0.8164966
#   through-origin slope fit = sqrt(18)/6 = 1/sqrt(2)
SLOPE_FIT_MSD_3D = math.sqrt(18.0) / 6.0             # 0.7071068

# 1-D variance (e.g. an axial dispersion coefficient), x^2/(2Dt) ~ chi^2_1.
#   relative sd at one time = sqrt(2*1)/1 = sqrt(2)
SINGLE_TIME_VARIANCE_1D = math.sqrt(2.0)             # 1.4142136
#   through-origin slope fit = sqrt(2)*sqrt(3)/2 = sqrt(6)/2
SLOPE_FIT_VARIANCE_1D = math.sqrt(6.0) / 2.0         # 1.2247449

LAWS = {
    "single_time_msd_3d": SINGLE_TIME_MSD_3D,
    "slope_fit_msd_3d": SLOPE_FIT_MSD_3D,
    "single_time_variance_1d": SINGLE_TIME_VARIANCE_1D,
    "slope_fit_variance_1d": SLOPE_FIT_VARIANCE_1D,
}


def sigma(law, n_particles):
    """Predicted relative standard error of `law`'s estimator at ensemble N."""
    if law not in LAWS:
        raise ValueError(f"unknown error law '{law}': expected one of {sorted(LAWS)}")
    return LAWS[law] / math.sqrt(n_particles)


def ks_critical(n_samples, alpha=0.05):
    """Asymptotic Kolmogorov-Smirnov critical value, ~1.36/sqrt(N) at 95%.

    This is the statistical FLOOR of the KS statistic: a sample drawn from the
    reference distribution sits at or below it. A value above the floor that
    FALLS with a numerical parameter is a real error being resolved, not noise.
    """
    coefficients = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}
    if alpha not in coefficients:
        raise ValueError(f"alpha must be one of {sorted(coefficients)}")
    return coefficients[alpha] / math.sqrt(n_samples)
