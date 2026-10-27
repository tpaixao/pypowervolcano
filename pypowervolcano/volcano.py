"""Power-aware volcano plots, model-agnostic by construction."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ._common import prepare
from .inversion import pvalue as _pvalue
from .power import p_at_power, power, power_bounds

__all__ = ["volcano"]


def volcano(
    estimate,
    se,
    pvalue=None,
    df=None,
    alpha=0.05,
    target_power=0.8,
    df_low=2.0,
    conservative_unknown=True,
    cmap="viridis",
    ax=None,
    colorbar=True,
    significance_line=True,
    annotate_counts=True,
    point_size=8,
    title=None,
    xlabel="effect size",
    **scatter_kwargs,
):
    """Power-aware volcano plot: x = effect, y = -log10 p, color = power.

    df handling defines the three regimes:

    - scalar or per-gene ``df`` (limma moderated t, edgeR QL,
      Satterthwaite): exact noncentral-t power per point. A horizontal
      frontier line at ``p_at_power(target_power, df)`` marks where
      plug-in power hits the target; per-gene df arrays use the median
      df for the reference line.
    - ``df=None`` (df genuinely unknown): points are colored by the
      interval from ``power_bounds``, the conservative lower bound
      (``power(df_low)``, evaluated at the known ncp = estimate/se) by
      default or the midpoint when ``conservative_unknown=False``.
      The target frontier becomes a shaded band between the z and
      df_low lines: above the band, power >= target under every df in
      [df_low, inf); below it, under target under every df; inside,
      the unknown df decides. Note the two regimes answer different
      questions and their orderings do not conflict: at a fixed
      ncp smaller df means less power (coloring), but at a fixed
      p-value smaller df means a larger statistic and more power
      (frontier lines).

    ``pvalue`` is the raw two-sided p-value from the same model as
    ``se``; when omitted it is recomputed under the z reference (or
    the given df), which is only faithful if the upstream test used
    the same reference.

    Returns ``(fig, ax)``.
    """
    est, se_arr, df_arr = prepare(estimate, se, df)
    if est.ndim != 1:
        raise ValueError("volcano expects 1-D inputs")

    if pvalue is None:
        p = _pvalue(est, se_arr, df)
    else:
        p = np.broadcast_arrays(np.asarray(pvalue, dtype=float), est)[0]
    y = -np.log10(np.clip(p, 1e-300, 1.0))

    if df_arr is None:
        lo, hi = power_bounds(est, se_arr, alpha=alpha, df_low=df_low)
        pw = lo if conservative_unknown else 0.5 * (lo + hi)
    else:
        pw = power(est, se_arr, df=df_arr, alpha=alpha)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    sc = ax.scatter(est, y, c=pw, cmap=cmap, vmin=0.0, vmax=1.0, s=point_size, **scatter_kwargs)

    # target-power frontier
    if df_arr is None:
        # at a fixed p-value, smaller df implies a larger statistic,
        # hence higher power: the df_low frontier sits lower on the
        # plot than the z frontier. Above the band, power >= target
        # under every df in [df_low, inf); below the band, under target
        # under every df; inside the band, the unknown df decides.
        p_low_df = p_at_power(target_power, df=df_low, alpha=alpha)
        p_z = p_at_power(target_power, df=None, alpha=alpha)
        y_low = -np.log10(p_low_df)
        y_high = -np.log10(p_z)
        ax.axhspan(y_low, y_high, color="0.5", alpha=0.12)
        ax.axhline(
            y_low,
            color="0.3",
            linestyle="--",
            linewidth=1,
            label=f"power={target_power:g} frontier at df={df_low:g}",
        )
        ax.axhline(
            y_high,
            color="0.3",
            linestyle=":",
            linewidth=1,
            label=f"power={target_power:g} frontier under z",
        )
    else:
        df_ref = float(np.median(df_arr)) if np.ndim(df_arr) else float(df_arr)
        p_t = p_at_power(target_power, df=df_ref, alpha=alpha)
        ax.axhline(
            -np.log10(p_t),
            color="0.3",
            linestyle="--",
            linewidth=1,
            label=f"power={target_power:g} (df={df_ref:g})",
        )

    if significance_line:
        ax.axhline(
            -np.log10(alpha),
            color="crimson",
            linestyle="-",
            linewidth=0.8,
            alpha=0.7,
            label=f"p = {alpha:g}",
        )

    if annotate_counts:
        n_sig = int(np.sum(p < alpha))
        n_pow = int(np.sum(pw >= target_power))
        ax.text(
            0.02,
            0.98,
            f"p < {alpha:g}: {n_sig}\npower >= {target_power:g}: {n_pow}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )

    if colorbar:
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("plug-in power")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("$-\\log_{10}$ p")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    return fig, ax