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
fixed-sample-size fitting functions. Malformed genotypes
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

These are marginal expectations under a neutral, constant-size equilibrium
model. They average over selfing histories and do not condition on those of a
fixed sample or define a joint multilocus distribution.

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

`support_interval` is descriptive likelihood support, not a calibrated confidence
interval. Shared individual selfing histories can induce dependence even across
unlinked loci. Metadata records
`uncertainty_calibration="uncalibrated_composite_likelihood"`.

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
example. Draws retain the same individual IDs across sites. Overlapping draws
are sensitivity analyses, not independent replicates; do not sum their likelihoods.

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
`selfdgs.plotting`. Summed likelihood and cumulative-convergence plots require
independent datasets at a common generating selfing rate. Nested samples and
overlapping draws do not meet that requirement.

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

These functions configure and run one simulation. Install SLiM 5.2 separately.
`default_slim_script_path()` selects the fixed-individual multilocus script;
pass `"independent_populations"` to select the single-locus control. The
multilocus script requires
`n_loci` and `history_path` defines, which the validation workflow manages. See
the [SLiM cookbook recipe](cookbook/03_slim_validation.ipynb).
Simulation sizes and seeds must be integers, rates must be finite and within
their documented ranges, and `extra_defines` keys must be valid Eidos
identifiers (see the SLiM manual for more details). 
Boolean define values are emitted as Eidos `T` and `F`. `None` and
non-finite numeric define values are rejected.

## Method validation

```python
from selfdgs.validation import (
    VCFResult,
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

`run_validation_experiment` defaults to a fixed sample across loci in one
shared-pedigree population per replicate:

```python
config = ValidationExperimentConfig(
    outdir="validation",
    sampling_design="fixed_individuals",
    census_size=100,
    n_sample=4,
    n_loci=30,
    chrom_length_each=20_000,
    mu=1e-5,
    n_reps=10,
)
replicates = run_validation_experiment(config)
```

`n_loci` specifies the number of loci under either design. Fixed-individual
runs require a positive
`burn_mult` and produce one VCF plus `sampled_individuals.csv` per replicate.
The manifest records zero-based VCF column, VCF sample ID, population index,
consecutive selfing generations, replicate, and seed. A history of `-1` is
unknown because no outcross was observed during the simulation.

Set `sampling_design="independent_populations"` to pool separate populations
and samples across loci as a marginal-expectation control. The script is
selected automatically when `slim_script` is omitted. A custom fixed-individual
script must emit one
multilocus VCF and a CSV at `history_path` with integer columns `vcf_column`,
`population_index`, and `selfing_generations` in VCF order.

`summarize_existing_vcfs` pools supplied VCFs. Its default fixed-individual mode
requires identical focal sample IDs across files (column order may differ).
Matching names alone cannot establish that the samples are biologically the same.
Use the independent-populations design only when different samples are intended.

The loading functions retain sampling design, locus count, and sample size.
`summarize_estimator_recovery` groups by these fields and by census size and true
selfing rate when available, reporting bias, MAE, and RMSE. Its `n_runs` counts
fitted rows, which need not represent independent populations in paired or nested
studies. Retain population replicate identities when comparing such conditions.

Simulation configuration uses `census_size` (default 500) and validation uses
`n_loci` (default 1). The Python and CLI validation workflows both default to
fixed individuals and folded polarization. Empirical analyses default to one
draw in both interfaces.

Each `ReplicateResult.vcf_results` entry is a `VCFResult` describing one VCF,
identified by `vcf_index`, with its `n_loci` count. The table is saved as
`vcfs.csv`. A fixed-individual replicate has one multilocus VCF; an
independent-populations replicate has one VCF per locus.

Accepted aliases are `ne` for `census_size`, `n_independent_loci` for `n_loci`,
and CLI `--ne` for `--census-size`. Supplying conflicting parameter aliases is
an error. `LocusResult`, `ReplicateResult.loci`, and `VCFResult.locus` are aliases
for `VCFResult`, `vcf_results`, and `vcf_index`, respectively. Saved tables with
an `ne` column are normalized to `census_size` for recovery summaries.
