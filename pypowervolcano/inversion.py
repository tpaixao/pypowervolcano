"""Recover the standard error from a raw p-value by inverting the test.

For any two-sided effect-scale test (Wald, t) the reported p-value
already encodes the SE: given the effect estimate theta and the raw
p-value, the observed statistic is ``q(1 - p/2)`` (normal or t
quantile), so ``SE = |theta| / q``. This is exact recovery, not an
approximation, provided the p-value is raw and comes from the same
effect-scale test.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._common import prepare

__all__ = ["se_from_pvalue", "pvalue"]


def pvalue(estimate, se, df=None):
    """Two-sided p-value of the statistic ``estimate / se``.

    Normal (z) when ``df`` is None, Student t with finite ``df``.
    """
    est, se_arr, df_arr = prepare(estimate, se, df)
    stat = est / se_arr
    if df_arr is None:
        return 2.0 * stats.norm.sf(np.abs(stat))
    return 2.0 * stats.t.sf(np.abs(stat), df_arr)


def se_from_pvalue(estimate, pvalue, df=None):
    """Standard error recovered by inverting a raw two-sided p-value.

    ``SE = |theta| / q_{1 - p/2}`` with q the normal (``df=None``) or
    t (``df``) quantile. Exact for raw p-values of effect-scale tests.

    Requirements
    ------------
    - RAW p-values: FDR-adjusted p-values break the inversion (the
      adjusted p is not the test tail probability).
    - Effect-scale tests only (Wald, t, moderated t). Rank-based
      tests must be rejected upstream: there ``|theta|`` is not the
      test statistic, so no SE lives in the p-value.
    - Finite ``df`` when the underlying test was a t: using the z
      quantile on t-test p-values INFLATES the recovered SE by the
      factor ``t*/z*`` (1.14x at df = 10, 1.31x at df = 5, 2.2x at
      df = 2, at alpha = 0.05), which is conservative in the
      noncentrality but silently wrong.

    Genes with ``estimate == 0`` or ``pvalue == 1`` return nan: the
    statistic is 0 by construction, so the SE is not recoverable from
    the p-value.
    """
    est = np.asarray(estimate, dtype=float)
    p = np.asarray(pvalue, dtype=float)
    est, p = np.broadcast_arrays(est, p)
    if np.any((p <= 0.0) | (p > 1.0)):
        raise ValueError("p-values must be raw and in (0, 1]")
    # isf(p/2) rather than ppf(1 - p/2): the latter loses all precision
    # for p below ~1e-16 (1 - p/2 rounds to 1 and the quantile to inf)
    if df is None:
        q = stats.norm.isf(p / 2.0)
    else:
        df_arr = np.broadcast_arrays(np.asarray(df, dtype=float), est)[0]
        q = stats.t.isf(p / 2.0, df_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.abs(est) / q
    return np.where((est == 0.0) | (p == 1.0), np.nan, se)