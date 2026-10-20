# powervolcano

Power-aware volcano plots for any effect-scale results table. Model-agnostic by construction: the package never sees a DESeq2, edgeR or limma object, it only needs the three numbers that determine statistical power: the effect estimate, its standard error, and the residual degrees of freedom.

Companion to the R package [volcano_power_plots](https://github.com/tpaixao/volcano_power_plots), which is DESeq2-specific; this one implements the generalized core and is deliberately adapter-free.

## The contract

Every function takes `(estimate, se, df)` arrays:

- `estimate`: effect estimate on any additive scale (log fold change, log hazard ratio, ...);
- `se`: its standard error, same units;
- `df`: residual degrees of freedom. `None` means the asymptotic (z) assumption; a finite value switches to the noncentral t family, exact for t statistics and recovering z as df grows.

## The math

Power is plug-in (conditional on the observed SE, a prediction, not a posterior):

```
power = P(|T| > t_{1 - alpha/2, df}),  T ~ noncentral t(df, ncp = estimate / se)
```

(the normal limit when `df is None`). All tail probabilities include both tails, so `power(0, se, ...) == alpha` exactly.

Key structural fact: with fixed `(df, alpha)` the statistic `|theta/se|` is sufficient for power, so on a volcano plot (x = effect, y = -log10 p) power is a monotone function of height alone and the power = target frontier is a horizontal line. `p_at_power()` returns the p-value of that line. This is what makes the whole thing model-agnostic: placing the frontier needs no per-gene plumbing, only `(df, alpha, target)`.

## When df is unknown

If the residual df is genuinely unknown, the z assumption is not silent: at alpha = 0.05 the critical value is 4.30 at df = 2 versus 1.96 under z, so the same effect/SE ratio that looks comfortably powered under z may be undetectable under t (at ncp = 2, z-based power is 0.52 versus 0.22 at df = 2). Instead of a point default, the package brackets the truth:

```python
from powervolcano import power_bounds, mde_bounds

lo, hi = power_bounds(theta, se)   # (power at df=2, power at z)
m_lo, m_hi = mde_bounds(se)        # (MDE at z, MDE at df=2)
```

In the volcano plot the unknown-df regime colors points by the conservative bound (power at df = 2, at the known ncp = estimate/se) and turns the target frontier into a shaded band between the z and df = 2 lines: above the band, power >= target under every df in [2, inf); below it, under target under every df; inside, the unknown df decides. One subtlety worth internalizing: at a fixed ncp smaller df means less power (that is the coloring bound), but at a fixed p-value smaller df means a larger statistic and therefore more power, which is why the df = 2 frontier sits lower on the plot than the z frontier.

## SE from a p-value

Any tool that reports an effect and a raw two-sided p-value has already told you the SE:

```python
from powervolcano import se_from_pvalue

se = se_from_pvalue(theta, p_raw, df=df)   # SE = |theta| / q_{1 - p/2}
```

Exact recovery, provided the p-value is raw (FDR-adjusted p breaks the inversion) and comes from an effect-scale test (Wald, t, moderated t). Rank-based tests must be rejected upstream: there the effect is not the test statistic, so no SE lives in the p-value. Genes with `estimate == 0` or `pvalue == 1` return nan. Two precision notes: the quantile is taken with `isf(p/2)`, not `ppf(1 - p/2)`, because the latter loses all precision for p below ~1e-16; and if the underlying test was a t but you invert with the z quantile, the recovered SE is inflated by the factor t*/z* (1.14x at df = 10, 2.2x at df = 2), which is silently conservative. DESeq2 is self-consistent here as a unit test: its Wald p-values are normal-based, so the recovered SE matches `lfcSE` to machine precision.

## Quickstart

```python
import numpy as np
from powervolcano import power, power_bounds, se_from_pvalue, mde, volcano

pw = power(theta, se, df=4)        # exact noncentral-t power, vectorized
lo, hi = power_bounds(theta, se)   # df unknown: [df=2, z] interval
se_rec = se_from_pvalue(theta, p_raw, df=4)
m = mde(se, df=4, target_power=0.8)

fig, ax = volcano(theta, se, pvalue=p_raw, df=4)      # frontier line
fig, ax = volcano(theta, se, pvalue=p_raw, df=None)   # frontier band
```

Per-gene df arrays work everywhere (limma moderated t df, edgeR QL denominator df, Satterthwaite), including as point colors and a median-df reference frontier in the plot.

## API

| function | what it does |
|---|---|
| `power(estimate, se, df, alpha)` | two-sided plug-in power, z or noncentral t |
| `power_bounds(estimate, se, df_low, df_high)` | interval when df is unknown |
| `mde(se, df, alpha, target_power)` | minimum detectable effect (exact brentq, not the z closed form) |
| `mde_bounds(se, ...)` | MDE interval when df is unknown |
| `p_at_power(target_power, df, alpha)` | p-value of the power frontier line |
| `power_table(estimate, se, ...)` | dict of arrays with power or bounds attached |
| `pvalue(estimate, se, df)` | two-sided p-value of the statistic |
| `se_from_pvalue(estimate, pvalue, df)` | SE recovered from a raw p-value |
| `volcano(estimate, se, pvalue, df, ...)` | the plot |

## Numerical notes

Two-sided t power is computed through the noncentral F (`|T| > t*` iff `T^2 > t*^2`, and `T^2 ~ F(1, df, ncp^2)`) rather than scipy's `nct`, which returns NaN for `|ncp|` beyond ~8 at small df. The `ncf.sf` route matches adaptive quadrature to ~1e-14 over the whole ncp range; it is broken exactly at `nc = 0`, where the answer is alpha by construction of the critical value, and that case is patched explicitly. All power values are exact at the boundary: `power(0, se, ...) == alpha`.

## Limitations

- Plug-in SE throughout: power conditional on the observed SE is a prediction, not a posterior statement.
- The inversion needs raw p-values from effect-scale tests.
- For paired or longitudinal designs the df (and the n behind prospective scaling) must come from the design; Satterthwaite or Kenward-Roger values plug in directly as per-gene df arrays.

## Development

```bash
uv venv .venv && uv pip install -e ".[test]" --python .venv/bin/python
.venv/bin/python -m pytest
```

MIT license.