"""Shared input validation helpers (internal)."""

from __future__ import annotations

import numpy as np


def prepare(estimate, se, df=None):
    """Broadcast and validate (estimate, se, df) inputs.

    Returns (estimate, se, df_arr) where df_arr is None when df is None
    (asymptotic / z case).
    """
    estimate, se = np.broadcast_arrays(
        np.asarray(estimate, dtype=float),
        np.asarray(se, dtype=float),
    )
    if np.any(~np.isfinite(se)) or np.any(se <= 0):
        raise ValueError("se must be positive and finite")
    if df is None:
        return estimate, se, None
    df_arr = np.broadcast_arrays(np.asarray(df, dtype=float), estimate)[0]
    if np.any(~np.isfinite(df_arr)) or np.any(df_arr <= 0):
        raise ValueError("df must be positive and finite")
    return estimate, se, df_arr


def check_alpha(alpha):
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def check_target_power(target_power, alpha):
    if not float(alpha) < float(target_power) < 1.0:
        raise ValueError("target_power must be in (alpha, 1)")