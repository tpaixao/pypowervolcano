# Methods

## Setup and notation

Per feature, the upstream model reported an effect estimate `theta`, its
standard error `se`, and (optionally) the residual degrees of freedom
`df`. The test statistic is `T = theta/se + noise`; under the null it is
standard normal (z reference) or Student t with `df`. `powervolcano` never
sees the underlying design matrix, count model, or likelihood: the three
numbers `(theta, se, df)` are the whole interface, which is what makes it
model-agnostic.

## Test-family assumptions

The `(theta, se, df)` interface encodes a Wald-style assumption: the null
distribution of the statistic `theta/se` must be standard normal (`df =
None`) or Student t with the residual df (finite df). The covered
families:

- Wald z-tests (the `df=None` case), including DESeq2 Wald tables, whose
  p-values are normal-based by construction;
- plain two-sample t-tests and Welch t-tests with the design df;
- limma moderated t with the residual df and, for a fitted eBayes
  object, the empirical-Bayes df (`df.prior + df.residual`) as a
  per-gene array;
- edgeR quasi-likelihood F-tests, since an `F(1, df)` is a squared t, so
  the same two-sided power machinery applies with the QL denominator df.

Explicitly excluded, because the effect is not proportional to the test
statistic: rank-based tests (Wilcoxon, Kolmogorov-Smirnov,
permutation-style rank tests), likelihood-ratio and score tests
referenced to a chi-square (the statistic is not `theta/se` scaled to a
t or z under chi-square), and any test whose p-value does not have
`theta/se` as its statistic. For these, `power` has no meaning and
`se_from_pvalue` silently produces garbage; reject them upstream.

Two further assumptions hold only approximately, and only with finite
df:

- one `(theta, se, df)` triple fully characterizes the null reference
  per feature (homoskedastic t-flavored behavior);
- theta is not rescaled independently of its SE. Shrunken estimators
  (apeglm-style log2FC shrinkage, where theta is shrunk but the SE is
  not, or vice versa) bias the plug-in noncentrality `theta/se`, so
  power values for shrunk tables are not trustworthy directly; run them
  on the unshrunken Wald table.


## Plug-in power

Power is evaluated conditional on the observed SE:

```
power(theta, se) = P(|T| > c),   T ~ t(df, ncp = theta/se),  c = t_{1-alpha/2, df}
```

(or the normal limit when `df is None`). This answers: *if the true effect
equals the observed estimate, and the SE is as observed, what is the
probability of rejection at level alpha?* It is a prediction, not a
posterior statement: it inherits the plug-in framing of volcanoPower, and
does not integrate over uncertainty in the SE or in the effect. Both tails
are included, so `power(0, se) == alpha` exactly.

## The noncentral-F route

scipy's `stats.nct` is numerically unreliable for two-sided power at
large noncentrality: it returns NaN for `|ncp|` beyond roughly 8 at small
df, exactly the regime where a power-aware volcano is most interesting
(strong effects, few residual df, e.g. 3 vs 3 designs). powervolcano
computes two-sided t power through the noncentral F distribution:

```
|T| > c  <=>  T^2 > c^2,   T^2 ~ F(1, df, ncp^2)
```

`ncf.sf(c^2, 1, df, ncp^2)` matches adaptive quadrature to ~1e-14 across
the whole ncp range. It is broken exactly at `ncp = 0`, where the answer
is alpha by construction of the critical value, and that case is patched
explicitly. The z case uses the closed two-tail normal sum, with the same
boundary patch.

## Power is a function of the statistic alone

The key structural fact: with fixed `(df, alpha)` the statistic
`|theta/se|` (the noncentrality) is *sufficient* for power. On a volcano
plot (x = effect, y = -log10 p) points with the same height have the same
|theta/se|, hence the same power: the power = target frontier is a
horizontal line, and its p-value depends only on `(df, alpha, target)`:

```
p_at_power(target) = 2 * tail(t*)  where t* solves power(t*) = target
```

No per-gene SE is needed to place the frontier; this is what allows the
annotation to be added to *any* results table without touching the fitted
model.

The numerical solve is exact (brentq on the noncentrality), unlike the
familiar MDE closed form `theta = se * (z_{1-alpha/2} + z_{target})`,
which is only the large-df limit and ignores the opposite tail.

## Unknown df: bracket, don't guess

If the residual df is genuinely unknown, assuming the z reference is not
a silent convenience: at alpha = 0.05 the critical value is 4.30 at df = 2
versus 1.96 under z, so the same effect/SE ratio that looks comfortably
powered under z may be undetectable under t (at ncp = 2: z power 0.52,
df = 2 power 0.22). z-based power therefore materially overstates
detectability exactly for the small-n designs where the question matters
(3 vs 3, residual df 4).

powervolcano brackets instead of guessing:

- **power**: `power_bounds(theta, se)` returns `[power(df_low), power(z)]`;
  the true power for any df in `[df_low, inf)` lies inside. Width of the
  interval = what the unknown df costs you.
- **MDE**: `mde_bounds(se)` returns `[mde(z), mde(df_low)]`, with the
  interpretation: an effect below the lower bound was undetectable even
  under the most favorable assumption; above the upper bound, detectable
  under the least favorable one.
- **plot**: points are colored by the conservative bound (power at
  df_low, evaluated at the known ncp = theta/se) and the frontier becomes
  a shaded band between the df_low and z lines:

```
        |        power >= target under every df in [df_low, inf)
  z line|
        |  band  the unknown df decides
 df=2   |
  line  |        power < target under every df
```

One subtlety worth internalizing: at a fixed ncp, smaller df means *less*
power (heavy tails, larger critical value; this is the coloring bound),
but at a fixed p-value, smaller df means a larger statistic and therefore
*more* power. That is why the df = 2 frontier line sits lower on the plot
than the z frontier line, and the band is wide rather than a point.

`df_low = 2` is a deliberately pessimistic default; pass the design's
actual minimal df if known (e.g. 4 for a 3 vs 3 comparison).

## Recovering SE from a p-value

Any tool that reports an effect and a raw two-sided p-value has already
told you the SE. Given `p`, the observed statistic solves
`p = 2 * tail(|T|)`, so `q = tail^{-1}(p/2)` and

```
se = |theta| / q
```

with `tail` the normal or t survival function matching the test family.
Exact recovery, subject to three conditions:

1. **raw p-values only.** Adjusted p-values (FDR/BH, Bonferroni) are not
   tail probabilities; inversion is meaningless.
2. **effect-scale tests only.** The effect must be the test statistic up
   to sign (Wald z, t, moderated t). For rank-based tests the effect is
   not the statistic, so no SE lives in the p-value and inversion
   silently produces garbage.
3. **right reference family.** Inverting t-test p-values with the z
   quantile inflates the recovered SE by the factor `t*/z*`: 1.14x at
   df = 10, 1.31x at df = 5, 2.2x at df = 2 (alpha = 0.05). Conservative
   in the noncentrality, but silently wrong; pass the df when you know
   it.

Numerics: the quantile is computed as `isf(p/2)`, not `ppf(1 - p/2)`,
which rounds to 1 (quantile = inf) for p below ~1e-16. Features with
`estimate == 0` or `p == 1` have a zero statistic by construction and
return nan.

DESeq2 Wald tables are self-consistent by construction (normal-based
p-values), so `se_from_pvalue(log2FoldChange, pvalue)` reproduces
`lfcSE` to machine precision; the test suite checks this on simulated
DESeq2-style output.

## Relation to prospective power

Prospective power (design a study, choose n) uses an anticipated effect
and the SE implied by n. Post hoc plug-in power uses the *observed* SE.
The two agree only when the observed SE equals the anticipated one; with
the observed data in hand, the SE is no longer hypothetical, so plug-in
power is the honest conditional statement a volcano plot can support.
Prospective scaling (what SE would achieve power 0.8 for this effect) is
available through `mde`: it inverts the question to the SE dimension.

## Limitations

- All power values are plug-in: conditional on the observed SE, a
  prediction, not a posterior.
- The SE inversion needs raw p-values from effect-scale tests.
- For paired or longitudinal designs the df (and the n behind
  prospective scaling) must come from the design; Satterthwaite or
  Kenward-Roger df plug in directly as per-gene arrays.
- Compositional effects, batch-confounded designs, or shrunken
  estimators change what `theta/se` means; the package annotates what
  the table says, not what the experiment was.