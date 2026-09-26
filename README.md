# selfdgs

`selfdgs` estimates population selfing rates from the diploid genotype
spectrum (DGS) of biallelic SNPs.

## Install

Install from source:

```bash
git clone https://github.com/pmckenz1/selfdgs.git
cd selfdgs
pip install .
```

## Fit a VCF

The default is a folded analysis, which is the best choice when ancestral alleles
are unknown:

```bash
selfdgs fit samples.vcf --out results/
```

Use `--polarization ref` only when VCF REF is independently known to be the
ancestral allele.

Start with `results/fit_summary.csv`:

- `best_s`: estimated selfing rate, from 0 to 1
- `support_interval`: likelihood support range (not automatically a calibrated
  confidence interval, even for unlinked loci)
- `n_sites`: number polymorphic sites used in the fit
- `warnings`: anything needing attention

The input may be VCF, compressed VCF, or BCF containing diploid, biallelic
SNPs. Parsing happens through `cyvcf2`.
Input files should have proper headers, including
definitions for referenced contigs and FORMAT fields. Only PASS biallelic SNPs
are retained by default, although you can change this with `--include-nonpass`.
Sites with missing genotypes are skipped by default so every
retained DGS cell has the specified sample size.
Malformed genotypes are always errors by default. Multiallelic sites are skipped.

## Choose polarization

| Available information | Option |
| --- | --- |
| Ancestral allele unknown (default) | `--polarization folded` |
| REF is known to be ancestral | `--polarization ref` |
| Reliable `INFO/AA` field | `--polarization aa` |
| Outgroup samples in the VCF | `--polarization outgroup-consensus --outgroup-samples ID1,ID2` |

We recommend a folded analysis unless you have a reliable way to identify the ancestral allele.

## Fixed individuals across loci

A genomic analysis should retain the same focal individuals at every site.
`selfdgs fit samples.vcf --out results/` uses all focal samples in the VCF;
`--n-diploids` checks their number and does not choose a subset. Missing calls
exclude the affected site rather than replacing an individual.

To select four named individuals from a larger VCF, put their IDs in
`focal_samples.txt`, one per line, and run:

```bash
selfdgs empirical samples.vcf \
  --sample-list focal=focal_samples.txt \
  --n-diploids 4 --n-draws 1 \
  --out fixed_sample_results/
```

The list must contain exactly four IDs to use all four without subsampling.
`draw_manifest.csv` records the selected individuals. Each draw holds its
individuals fixed across all retained sites.

The fitted model averages over individual selfing histories. It does not
condition on the realized histories of your sample. Shared histories can
induce dependence among unlinked loci, so adding loci to a small fixed sample
does not provide independent draws of those histories. Interpret likelihood
support as descriptive; assess estimator variation using independent population
replicates with a matching sampling design.

## Other common tasks

Assess sensitivity to sample choice with repeated fixed-size subsets:

```bash
selfdgs empirical samples.vcf \
  --n-diploids 4 \
  --n-draws 100 \
  --mode folded \
  --out results/
```

Run those draws separately within metadata-defined populations. Here,
`metadata.csv` has columns named `sample` (matching VCF sample IDs) and
`population`:

```bash
selfdgs empirical samples.vcf \
  --metadata metadata.csv \
  --group-column population \
  --n-diploids 4 \
  --n-draws 100 \
  --out grouped_results/
```

Fit on a specified grid and then continuously refine the best grid estimate
(omit `--refine` to report the grid estimate):

```bash
selfdgs fit samples.vcf \
  --grid 0:0.99:100 \
  --refine \
  --out refined_results/
```

Fit an existing DGS table:

```bash
selfdgs fit observed_dgs.csv --dgs-csv --out results/
```

Simulate a fixed sample across loci in a shared population pedigree:

```bash
selfdgs simulate --out validation/ --selfing-rate 0.5 \
  --sampling-design fixed_individuals --n-reps 10 \
  --census-size 100 \
  --n-loci 30 \
  --chrom-length 20000 \
  --mu 1e-5
```

Install SLiM 5.2 separately. Each replicate simulates one multilocus population,
then samples the same individuals across all loci. Adjacent loci have recombination
probability 0.5 between them. `--census-size` is the diploid census population size;
`--n-sample` is the number of sampled diploids. The default sampling design is
`fixed_individuals`.

Each `repNNN/` directory contains `sample.vcf`, `sampled_individuals.csv`, DGS
and likelihood tables, and SLiM logs. `vcfs.csv` records the VCF index, locus
count, seed, command, and simulation metadata. The sample manifest maps VCF columns to
population indices and consecutive selfing generations (`-1` means the history
is unknown because no outcross was observed). `validation_config.json` and
`validation_summary.csv` record the design and settings.

Use `--sampling-design independent_populations` to check marginal DGS
expectations by pooling a separate population and sample for each locus.
That control averages over individual histories and does not reproduce a
fixed-individual genomic dataset. See the [validation recipe](docs/cookbook/03_slim_validation.ipynb).

The simple `fit` vs. `empirical` commands use the same VCF scanning, filtering,
polarization, DGS construction, and likelihood code. The `empirical` command just
adds some potentially useful options like sample grouping and repeated fixed-size
sample draws. Overlapping draws measure sensitivity to sample choice; they are
not independent replicates and their likelihoods should not be summed.

## Read the outputs

- A single fit writes `fit_summary.csv`, `fit_result.json`, `likelihood.csv`,
  `observed_dgs.csv`, `fit_dgs.csv`, and `site_diagnostics.csv`. Start with
  `fit_summary.csv`
- For the grouped or repeated-draw analyses, start with
  `group_fit_summary.csv`, then check `sample_audit.csv` and
  `draw_diagnostics.csv` before interpreting estimates
- Use `fit_result.json` when code needs machine-readable support intervals,
  warnings, other nested metadata
- DGS tables use `n0`, `n1`, `n2`, and `count`. The first three columns count
  the first homozygous class, heterozygotes, and the second homozygous class.
  `count` is the number of sites with that configuration. The meaning of the
  two homozygous classes depends on the chosen polarization, with the two
  allele orientations being combined in a folded analysis.

When an output directory already exists, workflow-owned outputs from an earlier
run are replaced as needed and unrelated files are preserved.

## How it works

Each site is summarized by its counts of the three diploid genotype classes.
`selfdgs` compares the observed spectrum with its expectation under a
partial-selfing model and maximizes a composite likelihood. The likelihood is
conditional on segregating sites, excluding the two monomorphic configurations.

The model assumes a neutral, constant-size population at equilibrium under a
constant selfing rate. Its single-site probabilities average over selfing
histories; they do not specify the joint distribution across loci in fixed
individuals. Both physical linkage and shared histories can concentrate the
composite-likelihood curve without a corresponding gain in sampling precision.
LD pruning alone does not remove the shared-history dependence. The package
reports uncalibrated likelihood support ranges and does not implement a
calibrated confidence interval for this design.

## Frequently asked questions

### I have samples from a structured population. Can I run `selfdgs`?

Yes, but first assess population structure with STRUCTURE or a comparable
method. Then run `selfdgs` separately on each identified subpopulation.
Pooling differentiated subpopulations makes an apparent heterozygote deficit (the
Wahlund effect), so ignoring structure can inflate the
inferred selfing rate. Definitely interpret results cautiously when ancestry is admixed or
subpopulation assignments are unknown.

### What if I don't know the ancestral allele?

Use `--polarization folded`. Don't treat VCF REF as ancestral unless that
orientation has been established independently.

### How much data do I need?

There is no universal minimum. Both the number of individuals and the number
of retained polymorphic sites matter. More sites characterize the spectrum of
a fixed sample more closely, but do not add independent individual selfing
histories. A sharply peaked likelihood alone does not establish precision.
Inspect `n_sites`, warnings, and the curve, and use fixed-individual simulations
with independent population replicates to assess performance for your design.
Repeated subsets can help assess sensitivity to sample choice.

### Can I use linked SNPs?

Yes. Physical linkage adds dependence to the shared selfing histories of fixed
individuals. The reported support interval is uncalibrated for both linked and
unlinked loci.

### What should I do with missing, multiallelic, or non-diploid genotypes?

The VCF reader expects diploid, biallelic genotypes. By default, an entire site
is skipped if any requested sample has a missing genotype. Filter or recode unsupported records before
fitting, and review how much data was excluded so that missingness or filtering
does not silently determine the result.

## Future development

- Per-site variable sample size to better maximize information in the presence of missing data

## Documentation

- [Cookbook notebooks](docs/cookbook/README.md)
- [Python API](docs/api.md)

Run `selfdgs --help` or a subcommand's `--help` for command-line options.

Python 3.10 or newer is required.
