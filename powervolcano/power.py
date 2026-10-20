"""Core power computations for two-sided effect-scale tests.

The module is deliberately model-agnostic: the only inputs are the
effect estimate (theta), its standard error (SE), and the residual
degrees of freedom (df). ``df=None`` selects the asymptotic normal (z)
approximation; a finite ``df`` selects the noncentral t family, which
is exact for t-distributed test statistics and recovers the z case as
df grows.

All power values are plug-in: they are conditional on the observed SE,
i.e. they answer "if the true effect equals the observed estimate and
the SE is as observed, what is the probability of rejecting". This is
a prediction, not a posterior statement, and is inherited deliberately
from the volcanoPower design.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from ._common import check_alpha, check_target_power, prepare
from .inversion import pvalue as _pvalue

__all__ = [
    "power",
    "power_bounds",
    "mde",
    "mde_bounds",
    "p_at_power",
    "power_table",
]


def power(estimate, se, df=None, alpha=0.05):
    """Two-sided plug-in power at the observed effect size.

    Parameters
    ----------
    estimate : array_like
        Effect estimate (log fold change, log hazard ratio, ...), any
        additive scale.
    se : array_like
        Standard error of the estimate, same units.
    df : float, array_like or None
        Residual degrees of freedom. ``None`` means the test is
        asymptotic (z); a finite value uses the noncentral t family.
    alpha : float
        Two-sided significance level.

    Returns
    -------
    ndarray
        ``power = P(|T| > t_{1 - alpha/2, df})`` for
        ``T ~ noncentral t(df, ncp = estimate / se)``; the normal
        limit when ``df`` is None.
    """
    check_alpha(alpha)
    estimate, se, df_arr = prepare(estimate, se, df)
    ncp = estimate / se
    if df_arr is None:
        crit = stats.norm.ppf(1.0 - alpha / 2.0)
        pw = stats.norm.sf(crit - ncp) + stats.norm.cdf(-crit - ncp)
        # exact size at the boundary (the two-tail sum is 1 ulp off)
        return np.where(ncp == 0.0, alpha, pw)
    # Two-sided t power via the noncentral F: |T| > t* <=> T^2 > t*^2 and
    # T^2 ~ F(1, df, ncp^2). scipy's nct is numerically unreliable for
    # |ncp| beyond ~8 (NaNs); ncf.sf matches quadrature to ~1e-14 across
    # the whole range, but is broken exactly at nc = 0, where the answer
    # is alpha by construction of the critical value.
    crit = stats.t.ppf(1.0 - alpha / 2.0, df_arr)
    ncp2 = ncp ** 2
    pw = stats.ncf.sf(crit ** 2, 1, df_arr, ncp2)
    pw = np.where(ncp2 == 0.0, alpha, pw)
    return np.clip(pw, 0.0, 1.0)


def power_bounds(estimate, se, alpha=0.05, df_low=2.0, df_high=None):
    """Power interval when the residual df is genuinely unknown.

    Returns ``(power_low, power_high)``: plug-in power evaluated at
    ``df_low`` (conservative, heavy tails) and at ``df_high`` (``None``
    means infinity, the z limit). The true power for any unknown
    ``df`` in ``[df_low, df_high]`` lies inside the interval, so the
    width of the interval is exactly what the unknown df costs you.

    This matters most at small df: at alpha = 0.05 the critical value
    is 4.30 at df = 2 versus 1.96 under z, so an effect/SE ratio that
    looks comfortably detectable under z may be undetectable under t;
    z-based power materially overstates detectability exactly for the
    small-n designs people worry about (3 vs 3, residual df 4).
    """
    power_low = power(estimate, se, df=df_low, alpha=alpha)
    if df_high is None:
        power_high = power(estimate, se, df=None, alpha=alpha)
    else:
        power_high = power(estimate, se, df=df_high, alpha=alpha)
    return power_low, power_high


def mde(se, df=None, alpha=0.05, target_power=0.8):
    """Minimum detectable effect: smallest ``|theta|`` with power >= target.

    Solves ``power(theta, se, df, alpha) = target_power`` exactly
    (brentq on the noncentrality). The familiar closed form
    ``theta = se * (z_{1 - alpha/2} + z_{power})`` is the large-df
    limit and ignores the opposite tail; this function is exact.
    """
    check_alpha(alpha)
    check_target_power(target_power, alpha)
    scalar_input = np.ndim(se) == 0
    se_arr = np.atleast_1d(np.asarray(se, dtype=float))
    if np.any(~np.isfinite(se_arr)) or np.any(se_arr <= 0):
        raise ValueError("se must be positive and finite")
    if df is None:
        df_arr = None
    else:
        df_arr = np.broadcast_arrays(np.asarray(df, dtype=float), se_arr)[0]
    out = np.empty(se_arr.shape, dtype=float)
    se_flat = se_arr.ravel()
    df_flat = None if df_arr is None else df_arr.ravel()
    for i, s in enumerate(se_flat):
        d = None if df_flat is None else float(df_flat[i])
        crit = stats.norm.ppf(1.0 - alpha / 2.0) if d is None else stats.t.ppf(1.0 - alpha / 2.0, d)

        def f(theta, s=s, d=d):
            return float(power(theta, s, df=d, alpha=alpha)) - target_power

        hi = s * (crit + 60.0)
        out.ravel()[i] = optimize.brentq(f, 0.0, hi, xtol=1e-12, rtol=1e-10)
    return float(out.ravel()[0]) if scalar_input else out


def mde_bounds(se, alpha=0.05, target_power=0.8, df_low=2.0, df_high=None):
    """MDE interval when df is unknown.

    Returns ``(mde_low, mde_high)`` where the lower bound is the
    optimistic z limit (``df_high=None``) and the upper bound is the
    conservative ``df_low`` MDE. If the observed effect is below
    ``mde_low``, it was undetectable even under the most favorable
    assumption; if it exceeds ``mde_high``, it was detectable under
    the least favorable one.
    """
    if df_high is None:
        mde_low = mde(se, df=None, alpha=alpha, target_power=target_power)
    else:
        mde_low = mde(se, df=df_high, alpha=alpha, target_power=target_power)
    mde_high = mde(se, df=df_low, alpha=alpha, target_power=target_power)
    return mde_low, mde_high


def p_at_power(target_power, df=None, alpha=0.05):
    """The raw p-value at which plug-in power equals ``target_power``.

    With fixed (df, alpha) the statistic ``|theta/se|`` is sufficient
    for power, so on a volcano plot (x = effect, y = -log10 p) the
    power frontier is a horizontal line at this p-value. This is the
    key structural fact that makes power-aware volcano annotation
    model-agnostic: no per-gene SE is needed to place the frontier,
    only (df, alpha, target).
    """
    check_alpha(alpha)
    check_target_power(target_power, alpha)
    d = None if df is None else float(df)
    crit = stats.norm.ppf(1.0 - alpha / 2.0) if d is None else stats.t.ppf(1.0 - alpha / 2.0, d)

    def f(k):
        return float(power(k, 1.0, df=d, alpha=alpha)) - target_power

    k = optimize.brentq(f, crit, crit + 60.0, xtol=1e-12, rtol=1e-10)
    if d is None:
        return float(2.0 * stats.norm.sf(k))
    return float(2.0 * stats.t.sf(k, d))


def power_table(estimate, se, pvalue=None, df=None, alpha=0.05, df_low=2.0):
    """Convenience per-feature table (dict of arrays).

    Keys: ``estimate``, ``se``, ``pvalue`` (computed under the z
    reference if not supplied), and either ``power`` (df known) or
    ``power_low`` / ``power_high`` (df unknown).
    """
    est, se_arr, df_arr = prepare(estimate, se, df)
    if pvalue is None:
        p = _pvalue(est, se_arr, df)
    else:
        p = np.broadcast_arrays(np.asarray(pvalue, dtype=float), est)[0]
    out = {"estimate": est, "se": se_arr, "pvalue": p}
    if df is None:
        lo, hi = power_bounds(est, se_arr, alpha=alpha, df_low=df_low)
        out["power_low"] = lo
        out["power_high"] = hi
    else:
        out["power"] = power(est, se_arr, df=df_arr, alpha=alpha)
    return out