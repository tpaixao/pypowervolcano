import numpy as np
import pytest
from scipy import stats

from powervolcano import mde, mde_bounds, p_at_power, power, power_bounds, power_table

ALPHA = 0.05


def test_null_power_equals_alpha():
    # power at theta = 0 is the size of the two-sided test, exactly
    for df in (None, 2.0, 3.5, 10.0, 100.0):
        assert power(0.0, 1.0, df=df, alpha=ALPHA) == pytest.approx(ALPHA, abs=1e-7)


def test_power_monotone_in_effect():
    theta = np.linspace(0.0, 6.0, 50)
    for df in (None, 4.0):
        p = power(theta, 1.0, df=df)
        assert np.all(np.diff(p) >= 0)
        below_sat = p[:-1] < 1.0 - 1e-12
        assert np.all(np.diff(p)[below_sat] > 0)
        assert np.all((p >= ALPHA - 1e-12) & (p <= 1.0))


def test_t_power_le_z_power():
    # smaller df -> heavier tails -> lower power at the same ncp
    for th in (0.5, 1.0, 2.0, 4.0):
        pw_z = power(th, 1.0, df=None)
        for df in (2.0, 5.0, 10.0):
            assert power(th, 1.0, df=df) < pw_z


def test_power_bounds_bracket():
    lo, hi = power_bounds(2.0, 1.0)
    assert lo < hi
    for df in (2.0, 3.0, 5.0, 10.0, 50.0):
        mid = power(2.0, 1.0, df=df)
        assert lo <= mid <= hi


def test_power_bounds_collapse_with_df():
    lo, hi = power_bounds(2.0, 1.0, df_low=2.0)
    lo_big, hi_big = power_bounds(2.0, 1.0, df_low=200.0)
    assert (hi_big - lo_big) < (hi - lo)
    assert hi_big == pytest.approx(power(2.0, 1.0, df=None), abs=1e-9)


def test_mde_roundtrip():
    for df in (None, 2.0, 7.0, 30.0):
        for s in (0.1, 1.0, 2.5):
            m = mde(s, df=df)
            assert power(m, s, df=df) == pytest.approx(0.8, abs=1e-4)


def test_mde_z_closed_form():
    # large-df limit: theta = se * (z_{1-a/2} + z_{power}), opposite tail negligible
    expected = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(0.8)
    assert mde(1.0, df=None) == pytest.approx(expected, rel=1e-3)


def test_mde_grows_as_df_shrinks():
    m_z = mde(1.0, df=None)
    assert mde(1.0, df=2.0) > m_z
    assert mde(1.0, df=2.0) / m_z == pytest.approx(2.02, rel=0.05)


def test_mde_bounds_order():
    lo, hi = mde_bounds(1.0)
    assert lo < hi
    assert power(lo, 1.0, df=None) == pytest.approx(0.8, abs=1e-4)
    assert power(hi, 1.0, df=2.0) == pytest.approx(0.8, abs=1e-4)


def test_p_at_power_is_the_frontier():
    # the statistic at p_at_power must have exactly the target power
    for df in (None, 2.0, 5.0, 30.0):
        p_t = p_at_power(0.8, df=df)
        assert 0 < p_t < ALPHA
        if df is None:
            k = stats.norm.isf(p_t / 2)
        else:
            k = stats.t.isf(p_t / 2, df)
        assert power(k, 1.0, df=df) == pytest.approx(0.8, abs=1e-6)


def test_p_at_power_ordering():
    # at a fixed p-value, smaller df implies a LARGER statistic, hence
    # more plug-in power: the df=2 frontier sits at a larger p (lower
    # on the volcano) than the z frontier
    p2 = p_at_power(0.8, df=2.0)
    p10 = p_at_power(0.8, df=10.0)
    pz = p_at_power(0.8, df=None)
    assert p2 > p10 > pz


def test_power_table_known_df():
    theta = np.array([0.0, 1.0, 2.0])
    se = np.array([1.0, 1.0, 0.5])
    tab = power_table(theta, se, df=4.0)
    assert set(tab) == {"estimate", "se", "pvalue", "power"}
    np.testing.assert_allclose(tab["power"], power(theta, se, df=4.0))


def test_power_table_unknown_df():
    theta = np.array([0.0, 1.0, 2.0])
    se = np.array([1.0, 1.0, 0.5])
    tab = power_table(theta, se)
    assert "power_low" in tab and "power_high" in tab
    assert np.all(tab["power_low"] <= tab["power_high"])


def test_vectorization_matches_scalar():
    theta = np.array([0.3, 1.2, 2.7])
    se = np.array([0.4, 0.9, 0.2])
    for df in (None, 3.0):
        vec = power(theta, se, df=df)
        sca = [float(power(t, s, df=df)) for t, s in zip(theta, se)]
        np.testing.assert_allclose(vec, sca)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        power(1.0, -1.0)
    with pytest.raises(ValueError):
        power(1.0, 1.0, df=0)
    with pytest.raises(ValueError):
        power(1.0, 1.0, alpha=1.5)
    with pytest.raises(ValueError):
        mde(1.0, target_power=0.01)