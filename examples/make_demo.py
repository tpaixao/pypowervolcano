"""Demo figure: known df vs unknown df on the same simulated table.

Simulates a small-n two-group experiment (3 vs 3, residual df = 4)
with heterogeneous SEs, then renders the same volcano twice: exact
per-gene power under the known df, and the df-unknown regime where
points are colored by the conservative bound and the frontier becomes
a band.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pypowervolcano import volcano

rng = np.random.default_rng(42)
n = 800
is_de = rng.random(n) < 0.2
theta = np.where(is_de, rng.normal(2.5, 1.0, n), rng.normal(0.0, 0.4, n))
se = np.exp(rng.normal(-1.3, 0.35, n))
df = 4.0  # 3 vs 3, two-group design: residual df = n - 2
p = 2.0 * stats.t.sf(np.abs(theta / se), df)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
volcano(theta, se, pvalue=p, df=df, ax=axes[0],
        title="df known: 4 (3 vs 3), exact noncentral-t power")
volcano(theta, se, pvalue=p, df=None, ax=axes[1],
        title="df unknown: conservative bound, frontier band df=2 to z")
fig.tight_layout()
fig.savefig("examples/demo_volcano.png", dpi=150)
print("saved examples/demo_volcano.png")