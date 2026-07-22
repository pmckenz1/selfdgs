# Python API

The root `selfdgs` namespace contains the main functions for DGS inference.
Plotting, simulation, and validation tools live in separate subpackages.

For examples, start with the [README](../README.md) or
the [cookbook](cookbook/README.md).

## VCF and DGS data

```python
from selfdgs import (
    DGSError,
    DGSConfig,
    DGSCounts,
    VCFParseError,
    VCFScanResult,
    all_dgs_configs,
    counters_to_dataframe,
    dataframe_to_dgs,
    dgs_from_vcf,
    dgs_to_dataframe,
    fold_dgs_config,
    filter_polymorphic_dgs,
    is_monomorphic_config,
    read_dgs_csv,
    read_vcf_samples,
    scan_vcf_dgs,
    validate_dgs_config,
    validate_dgs_counts,
    write_dgs_csv,
)
```

`dgs_from_vcf` converts diploid, biallelic VCF or BCF records into DGS counts.
VCF/BCF parsing uses `cyvcf2`, while `selfdgs`
applies its own filtering, polarization, missing-data, and DGS rules. The
`polarization` modes are `ref`, `folded`, `aa`, and `outgroup-consensus`.
The default is `folded`. Use `ref` only when REF is independently known to be
ancestral.
Sites containing a missing genotype are excluded by default with
`missing="skip-site"`. Set `missing="error"` for strict input validation. The
`missing="skip-genotype"` option retains called individuals only and so
produces variable sample sizes. It is intended for inspection, not for the
fixed-sample-size fitting functions (although for future work we hope to extend
analysis to allow variable sample sizes across sites). Malformed genotypes
raise an error, and multiallelic records are excluded by default.

`scan_vcf_dgs` is the shared lower-level function used by simple and empirical
workflows. It accepts multiple named fixed sample sets, scans the VCF once, and
returns a `VCFScanResult` containing DGS counts plus
diagnostics. It retains PASS (or unset-FILTER) biallelic SNPs by default.
Variant files should have valid headers.

Use `filter_polymorphic_dgs` when looking at or exporting the segregating-site
spectrum. The fitting functions apply this conditioning automatically. Table
conversion and CSV helpers use the columns `n0`, `n1`, `n2`, and `count`.

## Analytical DGS functions

```python
from selfdgs import (
    dgs_probabilities,
    expected_dgs_branch_lengths,
    folded_dgs_probabilities,
    inbreeding_coefficient,
)
```

- `dgs_probabilities(s, N=1.0, n_diploids=4, method="combinatorial")`
  returns unfolded DGS probabilities.
- `folded_dgs_probabilities(...)` returns probabilities with the two allele
  orientations combined.
- `expected_dgs_branch_lengths(...)` returns the underlying expected branch
  lengths.
- `inbreeding_coefficient(s)` returns the equilibrium relationship
  `F = s / (2 - s)`.

## Likelihood fitting and results

```python
from selfdgs import (
    AllNonfiniteLikelihoodError,
    FitResult,
    LikelihoodPoint,
    default_selfing_grid,
    fit_result_from_json,
    fit_selfing,
    grid_search_selfing,
    likelihood_support_interval,
    log_likelihood_dgs,
    unsupported_dgs_cells,
)
```

`fit_selfing` is the main fitting starting point. It returns a `FitResult`
containing the best-scoring selfing rate, likelihood curve, support interval,
site counts, metadata, and warnings. `grid_search_selfing` provides the
grid-only fit used internally and for reproducible comparisons.

Use `FitResult.to_json`, `FitResult.to_csv`, `FitResult.summary_dict`, and
`fit_result_from_json` to save and load results.

## Empirical subset workflow

```python
from selfdgs import (
    EmpiricalAnalysisConfig,
    EmpiricalAnalysisResult,
    run_empirical_analysis,
)
```

`run_empirical_analysis` is the Python equivalent of
`selfdgs empirical`. It handles metadata-defined groups, fixed-sample-size random subset
draws, DGS construction, fitting, diagnostics, aggregation, and output
writing. Its default `missing="skip-draw"` excludes an affected site from each
draw containing a missing genotype while retaining the site for other complete
draws. So every fitted draw retains a fixed sample size. Use
`missing="skip-site"` to exclude a site with any missing data from all draws. See the
[fixed-size subset recipe](cookbook/02_subset_analysis.ipynb) for an
example.

The empirical workflow and `selfdgs fit` share the same VCF scanner and
likelihood implementation. A simple VCF fit writes a compact set of observed
and fitted DGS tables, fit results, likelihood values, and diagnostics.
The empirical layer adds grouping, repeated draws, aggregation, and plotting.

Lower-level functions are
available under `selfdgs.empirical`.

## Plotting

Import plotting functions from `selfdgs.plotting`:

```python
from selfdgs.plotting import (
    plot_best_s_distribution,
    plot_dgs_counts,
    plot_grouped_likelihood_summary,
    plot_joint_likelihood,
    plot_likelihood_curve,
    plot_likelihood_curves,
    plot_representative_observed_vs_fitted,
)
```

Validation-specific replicate and convergence plots also live in
`selfdgs.plotting`.

## Simulation

Using SLiM is optional and separate from inference:

```python
from selfdgs.simulation import (
    SLiMUnavailableError,
    SlimRunResult,
    SlimSimulationConfig,
    build_slim_command,
    default_slim_script_path,
    run_slim_simulation,
    slim_available,
)
```

These functions configure and run one simulation. Install SLiM separately. See
the [SLiM cookbook recipe](cookbook/03_slim_validation.ipynb).
Simulation sizes and seeds must be integers, rates must be finite and within
their documented ranges, and `extra_defines` keys must be valid Eidos
identifiers (see the SLiM manual for more details). 
Boolean define values are emitted as Eidos `T` and `F`. `None` and
non-finite numeric define values are rejected.

## Method validation

```python
from selfdgs.validation import (
    LocusResult,
    ReplicateResult,
    ValidationExperimentConfig,
    discover_validation_runs,
    load_likelihood_collection,
    load_validation_collection,
    run_validation_experiment,
    summarize_estimator_recovery,
    summarize_existing_vcfs,
)
```

`run_validation_experiment` pools simulated loci and fits replicate datasets.
The loading functions consolidate saved runs, while
`summarize_estimator_recovery` reports bias, MAE, and RMSE across known
simulation settings.
