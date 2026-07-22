# `selfdgs` Cookbook

The cookbook contains three executed Jupyter notebooks. You can see the
DGS tables, fit summaries, and plots in their outputs. The
notebooks can be rerun and modified locally.

## Recipes

1. [Quick folded fit](01_quick_fit.ipynb) generates a four-sample, 500-site
   pretend VCF at a known selfing rate, constructs a folded DGS, fits a
   selfing-rate grid, and plots the likelihood curve.
2. [Fixed-size subset analysis](02_subset_analysis.ipynb) generates a 12-sample,
   1,000-site pretend VCF with two simulated populations, then demonstrates 20
   reproducible draws per population, per-draw fits, summaries, and diagnostics.
3. [SLiM validation](03_slim_validation.ipynb) simulates and pools 30
   independent loci, fits the generating selfing rate, plots the likelihood,
   and explains how to consolidate experiments across generating settings.

## Run the notebooks

From the repository root, install the package with its development tools, then
start Jupyter:

```bash
python -m pip install -e ".[dev]"
jupyter lab docs/cookbook/
```

or non-interactively:

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  docs/cookbook/01_quick_fit.ipynb \
  docs/cookbook/02_subset_analysis.ipynb \
  docs/cookbook/03_slim_validation.ipynb
```