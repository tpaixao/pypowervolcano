# API reference

All functions operate elementwise and broadcast their inputs. Arrays
returned are numpy arrays; scalars in, scalars out where noted.

## `power(estimate, se, df=None, alpha=0.05)`

Two-sided plug-in power at the observed effect size.

- `estimate`: effect on any additive scale (log fold change, log hazard
  ratio, ...).
- `se`: standard error of the estimate, same units. Must be positive and
  finite.
- `df`: residual degrees of freedom. `None` selects the asymptotic normal
  (z) reference; a finite value (scalar or per-feature array) selects the
  noncentral t family. Per-gene df arrays are supported, so limma
  moderated-t df, edgeR QL denominator df, and Satterthwaite/Kenward-Roger
  df plug in directly.
- `alpha`: two-sided significance level, in (0, 1).

Returns `P(|T| > t_{1-alpha/2, df})` for `T ~ noncentral t(df,
ncp = estimate/se)`; the normal limit when `df is None`.

Exactness guarantees: `power(0, se, ...) == alpha` at the boundary, the
size of the test by construction. Computed through the noncentral F (see
methods) rather than `scipy.stats.nct`, which returns NaN for `|ncp|`
beyond roughly 8 at small df.

```python
from powervolcano import power
power(1.0, 0.5, df=4)       # 0.336
power(1.0, 0.5)             # 0.516 (z reference)
lo, hi = power_bounds(2.0, 0.9)   # (0.253, 0.603)
power(np.array([1., 2.]), np.array([0.5, 0.5]), df=4)
```

## `power_bounds(estimate, se, alpha=0.05, df_low=2.0, df_high=None)`

Power interval when the residual df is genuinely unknown. Returns
`(power_low, power_high)`: plug-in power at `df_low` (conservative, heavy
tails) and at `df_high` (`None` = infinity, the z limit). The true power
for any df in `[df_low, df_high]` lies inside the interval.

Use whenever the design df is not available (external tables, meta-analyses
of published results). At alpha = 0.05 the z-based power materially
overstates detectability for small-n designs: at ncp = 2, z power is 0.52
versus 0.22 at df = 2.

```python
from powervolcano import power_bounds
lo, hi = power_bounds(2.0, 0.9)   # (0.253, 0.603)
```

## `mde(se, df=None, alpha=0.05, target_power=0.8)`

Minimum detectable effect: smallest `|theta|` achieving `power >=
target_power` given the SE. Solved exactly with brentq on the
noncentrality; the familiar closed form
`theta = se * (z_{1-alpha/2} + z_{target})` is only the large-df limit and
ignores the opposite tail. `target_power` must lie in `(alpha, 1)`.

```python
from powervolcano import mde
mde(0.5, df=4, target_power=0.8)   # 1.881...
```

## `mde_bounds(se, alpha=0.05, target_power=0.8, df_low=2.0, df_high=None)`

MDE interval when df is unknown. Returns `(mde_low, mde_high)` where the
lower bound is the optimistic z-limit MDE and the upper bound the df_low
MDE. If the observed effect is below `mde_low`, it was undetectable even
under the most favorable assumption; above `mde_high`, detectable under
the least favorable one.

## `p_at_power(target_power, df=None, alpha=0.05)`

The raw p-value at which plug-in power equals `target_power`. With fixed
`(df, alpha)` the statistic `|theta/se|` is sufficient for power, so on a
volcano plot the power = target frontier is a horizontal line at this
p-value. This is what makes the frontier model-agnostic: placing it needs
no per-gene SE, only `(df, alpha, target)`.

```python
from powervolcano import p_at_power
p_at_power(0.8, df=4)    # 0.0198 at alpha=0.05
p_at_power(0.8, df=None) # 0.0051 (z frontier, higher on the plot)
```

## `pvalue(estimate, se, df=None)`

Two-sided p-value of the statistic `estimate/se`: normal when `df is
None`, Student t with finite df.

## `se_from_pvalue(estimate, pvalue, df=None)`

Standard error recovered by inverting a raw two-sided p-value:
`SE = |theta| / q_{1-p/2}`, with `q` the normal (`df=None`) or t (`df`)
quantile. Exact recovery, not an approximation.

Requirements:

- **raw** p-values. FDR-adjusted p-values break the inversion (the
  adjusted p is not the test tail probability).
- Effect-scale tests only (Wald, t, moderated t). Rank-based tests must
  be rejected upstream: there the effect is not the test statistic, so no
  SE lives in the p-value.
- Finite `df` when the underlying test was a t. Inverting with the z
  quantile inflates the recovered SE by the factor `t*/z*` (1.14x at
  df = 10, 1.31x at df = 5, 2.2x at df = 2 at alpha = 0.05): conservative
  in the noncentrality, but silently wrong.

Features with `estimate == 0` or `pvalue == 1` return nan (the statistic
is zero by construction, so the SE is not recoverable).

DESeq2 is self-consistent here: its Wald p-values are normal-based, so
`se_from_pvalue(log2FoldChange, pvalue)` reproduces `lfcSE` to machine
precision.

Implementation note: the quantile is taken with `isf(p/2)` rather than
`ppf(1 - p/2)`, which would lose all precision for p below ~1e-16.

## `power_table(estimate, se, pvalue=None, df=None, alpha=0.05, df_low=2.0)`

Convenience per-feature table as a dict of arrays. Keys: `estimate`,
`se`, `pvalue` (recomputed under the reference implied by `df` when not
supplied), and either `power` (df known) or `power_low` / `power_high`
(df unknown). Handy to attach power columns to a pandas DataFrame.

## `volcano(...)` — the plot

```
volcano(estimate, se, pvalue=None, df=None, alpha=0.05,
        target_power=0.8, df_low=2.0, conservative_unknown=True,
        cmap="viridis", ax=None, colorbar=True,
        significance_line=True, annotate_counts=True,
        point_size=8, title=None, xlabel="effect size",
        **scatter_kwargs)
```

Power-aware volcano plot: x = effect, y = -log10 p, color = plug-in
power. Returns `(fig, ax)`.

Three regimes, defined by the `df` argument:

- **df known** (scalar or per-feature array): exact noncentral-t power
  per point; horizontal frontier line at
  `p_at_power(target_power, df_ref)` where `df_ref` is the median of
  per-gene arrays. Above the line, plug-in power reaches the target.
- **df unknown** (`df=None`): points colored by the conservative lower
  bound `power(estimate, se, df=df_low)` (or the midpoint of the bounds
  when `conservative_unknown=False`); the target frontier becomes a
  shaded band between the df_low line and the z line. Above the band:
  power >= target under every df in `[df_low, inf)`. Below: under target
  under every df. Inside: the unknown df decides.

  Note the two orderings do not conflict: at a fixed ncp smaller df means
  *less* power (that is the coloring bound), but at a fixed p-value
  smaller df means a larger statistic and therefore *more* power, which
  is why the df=2 frontier sits lower on the plot than the z frontier.

Other arguments:

- `pvalue`: raw two-sided p-values from the same model as `se`. When
  omitted, p-values are recomputed under the reference implied by `df`,
  which is only faithful if the upstream test used the same reference.
- `significance_line`: optional line at `p = alpha`.
- `annotate_counts`: corner annotation of `p < alpha` and
  `power >= target_power` counts.
- `ax`: render into an existing axes (returns its figure).

## Errors raised

- `ValueError` if `se` is nonpositive or nonfinite, `df` is nonpositive
  or nonfinite, `alpha` outside (0, 1), `target_power` outside
  (alpha, 1), or p-values outside (0, 1].
- `volcano` additionally requires 1-D inputs.