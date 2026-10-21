# powervolcano docs

Power-aware volcano plots for any effect-scale results table. The package
needs exactly three inputs per feature: the effect estimate, its standard
error, and the residual degrees of freedom.

- **Getting started**: [index.md](index.md) (this page)
- **API reference**: [api.md](api.md)
- **Methods**: [methods.md](methods.md) (the math, the numerics, and what
  plug-in power does and does not mean)

## Installation

```bash
git clone https://github.com/tpaixao/powervolcano
cd powervolcano
uv venv .venv && uv pip install -e . --python .venv/bin/python
```

or from source in editable mode for development:

```bash
uv pip install -e ".[test]" --python .venv/bin/python
```

Dependencies: numpy, scipy, matplotlib (Python >= 3.10).

## Minimal tour

```python
import numpy as np
from powervolcano import power, power_bounds, mde, se_from_pvalue, volcano

power(2.0, 0.9, df=4)     # exact two-sided power, noncentral t
power_bounds(2.0, 0.9)    # df unknown: [power at df=2, power at z]
mde(0.9, df=4)            # smallest detectable effect at these SEs
se_from_pvalue(2.0, 3e-8) # SE recovered from a raw p-value
```

And the plot:

```python
theta, se, p = ...   # per-gene arrays from any DE table
fig, ax = volcano(theta, se, pvalue=p, df=4)       # df known
fig, ax = volcano(theta, se, pvalue=p, df=None)    # df unknown (band)
```

## Where it fits

powervolcano is the model-agnostic generalization of the R package
[volcano_power_plots](https://github.com/tpaixao/volcano_power_plots).

- R / volcanoPower: DESeq2-specific (LRT and Wald, per-gene df from the
  LRT denominator).
- Python / powervolcano: takes `(estimate, se, df)` and nothing else.
  DESeq2 Wald tables plug in directly (`log2FoldChange`, `lfcSE`, no df);
  limma moderated t, edgeR quasi-likelihood, and linear mixed models plug
  in their per-gene df (Satterthwaite or Kenward-Roger).