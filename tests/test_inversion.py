import numpy as np
import pytest
from scipy import stats

from pypowervolcano import pvalue, se_from_pvalue

RNG = np.random.default_rng(7)


def make_pair(n=200, df=None):
    theta = RNG.normal(0.0, 2.0, n)
    se = np.exp(RNG.normal(0.0, 0.5, n))
    stat = theta / se
    if df is None:
        p = 2.0 * stats.norm.sf(np.abs(stat))
    else:
        p = 2.0 * stats.t.sf(np.abs(stat), df)
    return theta, se, p


def test_z_roundtrip():
    theta, se, p = make_pair(df=None)
    rec = se_from_pvalue(theta, p, df=None)
    np.testing.assert_allclose(rec, se, rtol=1e-10)


def test_t_roundtrip():
    for df in (2.0, 4.0, 13.5):
        theta, se, p = make_pair(df=df)
        rec = se_from_pvalue(theta, p, df=df)
        np.testing.assert_allclose(rec, se, rtol=1e-10)


def test_z_inversion_of_t_pvalues_inflates_se():
    # the core small-df failure mode: z inversion inflates SE by t*/z*
    theta, se, p = make_pair(df=2.0)
    rec_z = se_from_pvalue(theta, p, df=None)
    bad = theta != 0
    assert np.all(rec_z[bad] > se[bad])
    assert np.median(rec_z[bad] / se[bad]) > 1.2


def test_pvalue_matches_manual():
    theta, se, _ = make_pair(df=None)
    np.testing.assert_allclose(pvalue(theta, se, df=None), 2.0 * stats.norm.sf(np.abs(theta / se)))
    theta, se, _ = make_pair(df=6.0)
    np.testing.assert_allclose(pvalue(theta, se, df=6.0), 2.0 * stats.t.sf(np.abs(theta / se), 6.0))


def test_bad_p_raises():
    with pytest.raises(ValueError):
        se_from_pvalue(1.0, 0.0)
    with pytest.raises(ValueError):
        se_from_pvalue(1.0, 1.5)
    with pytest.raises(ValueError):
        se_from_pvalue(np.array([1.0, 2.0]), np.array([0.5, -0.1]))


def test_zero_estimate_yields_nan():
    out = se_from_pvalue(0.0, 0.5, df=None)
    assert np.isnan(out)
    theta = np.array([0.0, 1.0])
    p = np.array([1.0, 0.3])
    out = se_from_pvalue(theta, p, df=None)
    assert np.isnan(out[0]) and np.isfinite(out[1])