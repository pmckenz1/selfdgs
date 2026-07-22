# selfdgs

`selfdgs` estimates population selfing rates from the diploid genotype
spectrum (DGS) of biallelic SNPs.

## Install

```bash
python -m pip install selfdgs
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
  confidence interval if there's linkage among sites)
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

## Other common tasks

Repeatedly fit randomly fixed-sample-size subsets from a larger VCF:

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

Run a validation experiment with the bundled SLiM script:

```bash
selfdgs simulate --out validation/ --selfing-rate 0.5 \
  --ne 100 \
  --n-loci 30 \
  --chrom-length 20000 \
  --mu 1e-5
```

SLiM must be installed separately.

The simple `fit` vs. `empirical` commands use the same VCF scanning, filtering,
polarization, DGS construction, and likelihood code. The `empirical` command just
adds some potentially useful options like sample grouping and repeated fixed-size
sample draws.

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

Linkage can make likelihood support intervals narrower than they should be.
It would be good to try calibrating uncertainty
with simulations or a bootstrap before interpreting a support interval
as a confidence interval, but we haven't implemented this yet.

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

There isn't a universal minimum. Informativeness depends more on the number of
retained polymorphic sites and how strongly the likelihood is peaked than on a
single sample-size cutoff. Check out `n_sites`, warnings, and the likelihood curve.
You can also use repeated fixed-size subsets to assess sensitivity to which individuals were
sampled, and you could use simulations to evaluate expected precision given some amount of data
in a power-analysis kind of approach.

### Can I use linked SNPs?

Yes, but linked sites are not independent. The estimate is based on a composite
likelihood, so the reported support interval can be too narrow when linkage is
ignored.

### What should I do with missing, multiallelic, or non-diploid genotypes?

The VCF reader expects diploid, biallelic genotypes. By default, an entire site
is skipped if any requested sample has a missing genotype. Filter or recode unsupported records before
fitting, and review how much data was excluded so that missingness or filtering
does not silently determine the result.

## Future development

- Per-site variable sample size to better maximize information in the presence of missing data

## Documentation

- [Executed cookbook notebooks](docs/cookbook/README.md)
- [Python API](docs/api.md)

Run `selfdgs --help` or a subcommand's `--help` for command-line options.

Python 3.10 or newer is required.
