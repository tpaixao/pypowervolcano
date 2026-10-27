import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

from pypowervolcano import volcano


@pytest.fixture()
def data():
    rng = np.random.default_rng(42)
    n = 400
    is_de = rng.random(n) < 0.2
    theta = np.where(is_de, rng.normal(2.5, 1.0, n), rng.normal(0.0, 0.4, n))
    se = np.exp(rng.normal(-1.3, 0.35, n))
    df = 4.0
    p = 2.0 * stats.t.sf(np.abs(theta / se), df)
    return theta, se, p


def test_volcano_known_df(data):
    theta, se, p = data
    fig, ax = volcano(theta, se, pvalue=p, df=4.0)
    assert fig is not None
    # one frontier line + one significance line, no band
    hlines = [ln for ln in ax.lines if ln.get_linestyle() in ("--", ":", "-")]
    assert len(hlines) >= 2
    assert len(ax.patches) == 0
    plt.close(fig)


def test_volcano_unknown_df_band(data):
    theta, se, p = data
    fig, ax = volcano(theta, se, pvalue=p, df=None)
    # the unknown-df regime draws the shading band
    assert len(ax.patches) == 1
    plt.close(fig)


def test_volcano_unknown_df_midpoint(data):
    theta, se, p = data
    fig, ax = volcano(theta, se, pvalue=p, df=None, conservative_unknown=False)
    assert fig is not None
    plt.close(fig)


def test_volcano_gene_wise_df(data):
    theta, se, p = data
    rng = np.random.default_rng(1)
    df = rng.integers(2, 30, size=theta.shape).astype(float)
    fig, ax = volcano(theta, se, pvalue=p, df=df)
    assert fig is not None
    plt.close(fig)


def test_volcano_computes_p_when_missing(data):
    theta, se, _ = data
    fig, ax = volcano(theta, se, df=4.0)
    assert fig is not None
    plt.close(fig)


def test_volcano_rejects_2d(data):
    theta, se, p = data
    with pytest.raises(ValueError):
        volcano(np.column_stack([theta, theta]), se, pvalue=p, df=4.0)