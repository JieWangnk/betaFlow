"""Monte Carlo error laws — one per ESTIMATOR, with derivations.

There is no "the Monte Carlo error law". There is the law for THIS estimator
on THIS fit, and the four below differ by up to a factor of two. Using one
where another belongs is the statistical analogue of reusing a channel analytic reference
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

    NOTE ON WHAT "AT FLOOR" LOOKS LIKE. 1.36/sqrt(N) is the 95th PERCENTILE,
    not the typical value: a correct sampler sits near the MEDIAN, which is
    about 0.83/sqrt(N), i.e. ~0.61 of this figure. So KS/floor landing
    consistently in 0.6-0.9 is exactly what drawing from the true
    distribution looks like. Sitting AT 1.0 would be suspicious, not ideal.
    """
    coefficients = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}
    if alpha not in coefficients:
        raise ValueError(f"alpha must be one of {sorted(coefficients)}")
    return coefficients[alpha] / math.sqrt(n_samples)


def binomial_sigma(p, n_particles):
    """ABSOLUTE standard deviation of a measured fraction, sqrt(p(1-p)/N).

    The channel-impulse-response value at one time is a binomial proportion:
    each of N released particles is inside the receiver window or not. Unlike
    the four relative laws above, this one depends on the value p itself and
    is returned in absolute units of the fraction, because relative error
    diverges as p -> 0 in the tail where the CIR is smallest.
    """
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p = {p} is not a probability")
    return math.sqrt(p * (1.0 - p) / n_particles)


def binomial_rmse_floor(p_curve, n_particles):
    """Expected RMSE of a measured fraction curve against its exact curve.

    E[(phat(t) - p(t))^2] = p(t)(1-p(t))/N exactly at every t, whatever the
    correlation between times, so the expected SQUARED RMSE is the time-mean
    of that and the floor is sqrt(mean(p(1-p))/N).

    CAVEAT the assertion must respect: one particle contributes to phat at
    EVERY time, so the errors at nearby times are strongly correlated and the
    measured RMSE scatters around this floor with O(1) relative spread — the
    effective number of independent time samples is the number of
    non-overlapping receiver windows in speed space, not the grid size. A
    fixed-seed check therefore asserts a BAND around the floor, not a
    one-sided multiple of a concentration bound that does not apply.
    """
    import numpy as np

    p = np.asarray(p_curve, dtype=float)
    return float(np.sqrt(np.mean(p * (1.0 - p)) / n_particles))


def binomial_integral_sigma_bound(times, p_curve, n_particles):
    """UPPER bound on the sd of the time-integral of a measured fraction.

    sd(integral phat dt) <= integral sd(phat) dt = integral sqrt(p(1-p)/N) dt,
    with equality when the errors are perfectly correlated across times
    (Minkowski's integral inequality). The tail-mass check uses this bound
    because the true covariance across times depends on the window overlap
    structure; the bound costs at most the correlation factor, still falls as
    1/sqrt(N), and can only make the assertion conservative.
    """
    import numpy as np

    t = np.asarray(times, dtype=float)
    p = np.asarray(p_curve, dtype=float)
    return float(np.trapezoid(np.sqrt(p * (1.0 - p) / n_particles), t))
