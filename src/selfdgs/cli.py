"""Command-line interface for selfdgs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Sequence

import pandas as pd

from selfdgs import __version__
from selfdgs._outputs import install_staged_outputs
from selfdgs.empirical import (
    DRAW_MANIFEST_COLUMNS,
    EMPIRICAL_OUTPUT_NAMES,
    SITE_DIAGNOSTICS_COLUMNS,
    EmpiricalAnalysisConfig,
    dgs_for_draws,
    make_draw_manifest,
    make_sample_groups,
    run_empirical_analysis,
)
from selfdgs.fit import fit_selfing
from selfdgs.io.vcf import dgs_from_vcf, read_vcf_samples
from selfdgs.model import dgs_probabilities, expected_dgs_branch_lengths
from selfdgs.spectrum import filter_polymorphic_dgs, read_dgs_csv, write_dgs_csv
from selfdgs.validation import ValidationExperimentConfig, run_validation_experiment

FIT_OUTPUT_NAMES = frozenset(
    {
        "observed_dgs.csv",
        "fit_dgs.csv",
        "fit_result.json",
        "likelihood.csv",
        "fit_summary.csv",
        "site_diagnostics.csv",
    }
)


def _infer_n_diploids(counts) -> int:
    totals = {sum(config) for config in counts}
    if not totals:
        raise ValueError("cannot infer n_diploids from empty DGS counts; pass --n-diploids.")
    if len(totals) != 1:
        raise ValueError(
            "cannot infer n_diploids from DGS counts with inconsistent cell totals; "
            "pass --n-diploids."
        )
    return totals.pop()


def _parse_grid(spec: str | None) -> list[float] | None:
    if spec is None:
        return None
    try:
        if ":" in spec:
            parts = spec.split(":")
            if len(parts) != 3:
                raise ValueError("grid range must have form start:stop:num.")
            start = float(parts[0])
            stop = float(parts[1])
            num = int(parts[2])
            if num < 1:
                raise ValueError("grid range num must be at least 1.")
            if num == 1:
                return [start]
            step = (stop - start) / (num - 1)
            return [start + i * step for i in range(num)]
        values = [float(value) for value in spec.split(",") if value.strip()]
    except ValueError as exc:
        if str(exc).startswith("grid range"):
            raise
        raise ValueError(
            "grid must be a comma-separated list of floats or start:stop:num."
        ) from exc
    if not values:
        raise ValueError("grid must contain at least one selfing-rate value.")
    return values


def _add_vcf_options(
    parser: argparse.ArgumentParser,
    *,
    include_n_diploids: bool = True,
    allow_skip_genotype: bool = True,
) -> None:
    if include_n_diploids:
        parser.add_argument(
            "--n-diploids",
            type=int,
            default=None,
            help="Number of diploid samples to read from the VCF.",
        )
    parser.add_argument(
        "--exclude-monomorphic",
        action="store_true",
        help="Exclude all-REF and all-ALT homozygous sites.",
    )
    genotype_policy_choices = ["error", "skip-site"]
    if allow_skip_genotype:
        genotype_policy_choices.append("skip-genotype")
    parser.add_argument(
        "--missing",
        choices=genotype_policy_choices,
        default="skip-site",
        help="Policy for VCF sites with missing genotypes (default: skip-site).",
    )
    parser.add_argument(
        "--malformed",
        choices=genotype_policy_choices,
        default="error",
        help="Policy for malformed or non-diploid genotypes.",
    )
    parser.add_argument(
        "--multiallelic",
        choices=["skip", "error"],
        default="skip",
        help="Policy for multiallelic VCF records.",
    )
    parser.add_argument(
        "--include-nonpass",
        action="store_true",
        help="Include VCF records whose FILTER value is not PASS or '.'.",
    )
    parser.add_argument(
        "--polarization",
        choices=["ref", "folded", "aa", "outgroup-consensus"],
        default="folded",
        help=(
            "Ancestral-state strategy for VCF-to-DGS conversion (default: "
            "folded). Use ref only when REF is independently known to be ancestral."
        ),
    )
    parser.add_argument(
        "--outgroup-samples",
        default=None,
        help="Comma-separated outgroup sample IDs for outgroup-consensus polarization.",
    )
    parser.add_argument(
        "--min-outgroup-called",
        type=int,
        default=1,
        help="Minimum called outgroup genotypes required for outgroup-consensus polarization.",
    )
    parser.add_argument(
        "--allow-heterozygous-outgroup",
        action="store_true",
        help=(
            "Allow heterozygous outgroup genotypes and infer the majority allele "
            "when the outgroup is polymorphic."
        ),
    )
    parser.add_argument(
        "--allow-polymorphic-outgroup",
        action="store_true",
        help="Allow polymorphic outgroups and use the majority allele as ancestral.",
    )


def _parse_sample_list(spec: str | None) -> list[str] | None:
    if spec is None:
        return None
    values = [value.strip() for value in spec.split(",") if value.strip()]
    return values or None


def _read_sample_list_file(path: str | Path) -> list[str]:
    samples: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        samples.extend(value.strip() for value in line.replace(",", " ").split() if value.strip())
    if not samples:
        raise ValueError(f"Sample list file {path!s} does not contain any sample IDs.")
    return samples


def _parse_sample_list_files(specs: Sequence[str], vcf_samples: Sequence[str]) -> dict[str, list[str]]:
    sample_set = set(vcf_samples)
    groups: dict[str, list[str]] = {}
    for spec in specs:
        if "=" in spec:
            group, path = spec.split("=", 1)
            group = group.strip()
            path = path.strip()
            if not group:
                raise ValueError(f"Sample list specification {spec!r} has an empty group name.")
        else:
            path = spec
            group = Path(path).stem
        samples = _read_sample_list_file(path)
        if group in groups:
            raise ValueError(f"Sample-list group name {group!r} is repeated.")
        if len(samples) != len(set(samples)):
            raise ValueError(f"Sample list {path!s} contains duplicate sample IDs.")
        missing = [sample for sample in samples if sample not in sample_set]
        if missing:
            raise ValueError(f"Sample list {path!s} contains samples absent from VCF: {missing[:10]}")
        groups[group] = samples
    return groups


def _sample_audit_from_groups(
    groups: dict[str, list[str]],
    *,
    min_samples: int,
) -> pd.DataFrame:
    rows = []
    for group, samples in sorted(groups.items()):
        n_samples = len(samples)
        rows.append(
            {
                "group": group,
                "n_samples": n_samples,
                "metadata_n": n_samples,
                "in_vcf_n": n_samples,
                "missing_from_vcf_n": 0,
                "included": n_samples >= min_samples,
            }
        )
    return pd.DataFrame(rows)


def _load_empirical_metadata(args: argparse.Namespace) -> pd.DataFrame | None:
    if args.metadata is None:
        return None
    metadata = pd.read_csv(args.metadata, dtype=str)
    for column in metadata.columns:
        metadata[column] = metadata[column].str.strip()
    return metadata


def _outgroup_samples_from_metadata(
    metadata: pd.DataFrame | None,
    *,
    sample_column: str,
    outgroup_column: str | None,
    outgroup_value: str | None,
    vcf_samples: Sequence[str],
) -> list[str]:
    if outgroup_column is None and outgroup_value is None:
        return []
    if metadata is None:
        raise ValueError("--metadata is required with --outgroup-column/--outgroup-value.")
    if not outgroup_column or outgroup_value is None:
        raise ValueError("--outgroup-column and --outgroup-value must be used together.")
    missing_columns = {sample_column, outgroup_column}.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")
    vcf_sample_set = set(vcf_samples)
    outgroup = [
        sample
        for sample in metadata.loc[metadata[outgroup_column] == outgroup_value, sample_column]
        .astype(str)
        .tolist()
        if sample in vcf_sample_set
    ]
    if not outgroup:
        raise ValueError(
            f"No VCF samples matched outgroup {outgroup_column}={outgroup_value!r}."
        )
    return outgroup


def _dgs_command(args: argparse.Namespace) -> int:
    counts = dgs_from_vcf(
        args.vcf,
        n_diploids=args.n_diploids,
        include_monomorphic=not args.exclude_monomorphic,
        missing=args.missing,
        malformed=args.malformed,
        multiallelic=args.multiallelic,
        polarization=args.polarization,
        outgroup_sample_ids=_parse_sample_list(args.outgroup_samples),
        min_outgroup_called=args.min_outgroup_called,
        require_homozygous=not args.allow_heterozygous_outgroup,
        skip_polymorphic_outgroup=not (
            args.allow_polymorphic_outgroup or args.allow_heterozygous_outgroup
        ),
        require_pass=not args.include_nonpass,
    )
    write_dgs_csv(counts, args.out)
    return 0


def _single_vcf_draw(args: argparse.Namespace):
    """Scan a VCF as one deterministic all-focal-samples draw."""
    vcf_samples = read_vcf_samples(args.input)
    outgroup_samples = _parse_sample_list(args.outgroup_samples) or []
    outgroup_set = (
        set(outgroup_samples)
        if args.polarization == "outgroup-consensus"
        else set()
    )
    focal_samples = [sample for sample in vcf_samples if sample not in outgroup_set]
    if args.n_diploids is not None and len(focal_samples) != args.n_diploids:
        raise ValueError(
            f"VCF header has {len(focal_samples)} samples, expected {args.n_diploids}."
        )
    n_diploids = len(focal_samples)
    manifest = pd.DataFrame(
        [
            {
                "group": "all_samples",
                "draw_id": 0,
                "n_diploids": n_diploids,
                "seed": 0,
                "sample_ids": ";".join(focal_samples),
            }
        ],
        columns=DRAW_MANIFEST_COLUMNS,
    )
    mode = "unfolded" if args.polarization == "ref" else args.polarization
    counts_by_draw, site_diagnostics, _ = dgs_for_draws(
        args.input,
        manifest,
        mode=mode,
        polarization=args.polarization,
        outgroup_sample_ids=outgroup_samples,
        min_outgroup_called=args.min_outgroup_called,
        require_homozygous=not args.allow_heterozygous_outgroup,
        skip_polymorphic_outgroup=not (
            args.allow_polymorphic_outgroup or args.allow_heterozygous_outgroup
        ),
        require_pass=not args.include_nonpass,
        missing=args.missing,
        malformed=args.malformed,
        multiallelic=args.multiallelic,
        include_monomorphic=not args.exclude_monomorphic,
    )
    return counts_by_draw[("all_samples", 0)], n_diploids, site_diagnostics


def _single_dgs_csv_draw(args: argparse.Namespace):
    """Read one DGS CSV for the compact single-fit workflow."""
    counts = read_dgs_csv(args.input)
    n_diploids = (
        args.n_diploids
        if args.n_diploids is not None
        else _infer_n_diploids(counts)
    )
    site_diagnostics = pd.DataFrame(
        [{"metric": "input_dgs_csv", "value": 1}],
        columns=SITE_DIAGNOSTICS_COLUMNS,
    )
    return counts, n_diploids, site_diagnostics


def _fit_command(args: argparse.Namespace) -> int:
    outdir = Path(args.out)

    counts, n_diploids, site_diagnostics = (
        _single_dgs_csv_draw(args) if args.dgs_csv else _single_vcf_draw(args)
    )

    fit_counts = filter_polymorphic_dgs(counts, n_diploids=n_diploids)
    grid = _parse_grid(args.grid)
    mode = args.mode or ("folded" if args.polarization == "folded" else "unfolded")
    result = fit_selfing(
        counts,
        n_diploids=n_diploids,
        s_grid=grid,
        epsilon=args.epsilon,
        error_rate=args.error_rate,
        support_drop=args.support_drop,
        refine=args.refine,
        refine_bounds=(args.refine_lower, args.refine_upper),
        mode=mode,
        input_polarization=args.polarization,
    )

    outdir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{outdir.name}.", dir=outdir.parent) as tempdir:
        staged_outdir = Path(tempdir) / "output"
        staged_outdir.mkdir()
        write_dgs_csv(counts, staged_outdir / "observed_dgs.csv")
        write_dgs_csv(fit_counts, staged_outdir / "fit_dgs.csv")
        result.to_json(staged_outdir / "fit_result.json")
        result.to_csv(staged_outdir / "likelihood.csv")
        pd.DataFrame([result.summary_dict()]).to_csv(
            staged_outdir / "fit_summary.csv",
            index=False,
        )
        site_diagnostics.to_csv(staged_outdir / "site_diagnostics.csv", index=False)
        install_staged_outputs(
            staged_outdir,
            outdir,
            owned_names=FIT_OUTPUT_NAMES,
        )
    return 0


def _empirical_command(args: argparse.Namespace) -> int:
    outdir = Path(args.out)

    vcf_samples = read_vcf_samples(args.vcf)
    metadata = _load_empirical_metadata(args)
    outgroup_samples = _parse_sample_list(args.outgroup_samples) or []
    outgroup_samples.extend(
        sample
        for sample in _outgroup_samples_from_metadata(
            metadata,
            sample_column=args.sample_column,
            outgroup_column=args.outgroup_column,
            outgroup_value=args.outgroup_value,
            vcf_samples=vcf_samples,
        )
        if sample not in outgroup_samples
    )
    active_outgroup_samples = (
        outgroup_samples if args.mode == "outgroup-consensus" else []
    )

    if args.sample_list:
        groups = _parse_sample_list_files(args.sample_list, vcf_samples)
        overlapping = {
            group: sorted(set(samples).intersection(active_outgroup_samples))
            for group, samples in groups.items()
        }
        overlapping = {
            group: samples for group, samples in overlapping.items() if samples
        }
        if overlapping:
            raise ValueError(
                "Explicit sample groups contain outgroup samples that cannot also "
                f"be focal: {overlapping}"
            )
        sample_audit = _sample_audit_from_groups(groups, min_samples=args.n_diploids)
    else:
        grouping = args.grouping
        if grouping == "auto":
            grouping = "grouped" if args.metadata and args.group_column else "all"
        grouping_metadata = metadata
        if (
            grouping in {"grouped", "both"}
            and metadata is not None
            and args.outgroup_column
            and args.outgroup_value is not None
        ):
            grouping_metadata = metadata.loc[metadata[args.outgroup_column] != args.outgroup_value]
        if (
            grouping_metadata is not None
            and active_outgroup_samples
            and args.sample_column in grouping_metadata.columns
        ):
            grouping_metadata = grouping_metadata.loc[
                ~grouping_metadata[args.sample_column].isin(active_outgroup_samples)
            ]
        focal_vcf_samples = [
            sample
            for sample in vcf_samples
            if sample not in set(active_outgroup_samples)
        ]
        sample_audit, groups = make_sample_groups(
            focal_vcf_samples,
            mode=grouping,
            metadata=grouping_metadata,
            sample_column=args.sample_column,
            group_column=args.group_column,
            min_samples=args.n_diploids,
            all_group_name=args.all_group_name,
        )

    too_small = sample_audit.loc[sample_audit["included"] == False]  # noqa: E712
    if not too_small.empty:
        details = ", ".join(
            f"{row.group} ({int(row.n_samples)}/{args.n_diploids})"
            for row in too_small.itertuples(index=False)
        )
        raise ValueError(f"Sample groups with too few VCF samples: {details}")

    if not groups:
        raise ValueError(f"No sample groups contain at least {args.n_diploids} VCF samples.")

    draw_manifest = make_draw_manifest(
        groups,
        n_diploids=args.n_diploids,
        n_draws=args.n_draws,
        seed=args.seed,
    )

    outdir.parent.mkdir(parents=True, exist_ok=True)
    temporary = TemporaryDirectory(prefix=f".{outdir.name}.", dir=outdir.parent)
    staged_outdir = Path(temporary.name) / "output"
    staged_outdir.mkdir()

    if args.audit_only:
        sample_audit.to_csv(staged_outdir / "sample_audit.csv", index=False)
        draw_manifest.to_csv(staged_outdir / "draw_manifest.csv", index=False)
        install_staged_outputs(
            staged_outdir,
            outdir,
            owned_names={"sample_audit.csv", "draw_manifest.csv"},
        )
        temporary.cleanup()
        print(
            f"Audit complete: {len(groups)} groups, {len(draw_manifest)} draws.",
            file=sys.stderr,
        )
        return 0

    if args.mode == "outgroup-consensus" and not outgroup_samples:
        raise ValueError(
            "outgroup-consensus mode requires --outgroup-samples or "
            "--metadata with --outgroup-column/--outgroup-value."
        )

    result = run_empirical_analysis(
        EmpiricalAnalysisConfig(
            vcf_path=args.vcf,
            n_diploids=args.n_diploids,
            n_draws=args.n_draws,
            seed=args.seed,
            mode=args.mode,
            fit_mode=(
                args.fit_mode
                or ("folded" if args.mode == "folded" else "unfolded")
            ),
            outgroup_sample_ids=active_outgroup_samples,
            min_outgroup_called=args.min_outgroup_called,
            require_homozygous=not args.allow_heterozygous_outgroup,
            skip_polymorphic_outgroup=not (
                args.allow_polymorphic_outgroup or args.allow_heterozygous_outgroup
            ),
            require_pass=not args.include_nonpass,
            max_records=args.max_records,
            missing=args.missing,
            malformed=args.malformed,
            multiallelic=args.multiallelic,
            s_grid=_parse_grid(args.grid),
            epsilon=args.epsilon,
            error_rate=args.error_rate,
            support_drop=args.support_drop,
            low_polymorphic_site_threshold=(
                None
                if args.low_polymorphic_site_threshold < 0
                else args.low_polymorphic_site_threshold
            ),
            outdir=staged_outdir,
        ),
        sample_groups=groups,
        sample_audit=sample_audit,
    )

    if args.plots:
        from selfdgs.plotting import (
            plot_best_s_distribution,
            plot_grouped_likelihood_summary,
            plot_representative_observed_vs_fitted,
        )

        plot_grouped_likelihood_summary(
            result.likelihood_curves,
            out=staged_outdir / "grouped_likelihood_summary.png",
        )
        plot_best_s_distribution(
            result.fit_summary,
            out=staged_outdir / "best_s_distribution.png",
        )
        plot_representative_observed_vs_fitted(
            result.representative_observed_vs_fitted,
            out=staged_outdir / "representative_observed_vs_fitted.png",
        )

    install_staged_outputs(
        staged_outdir,
        outdir,
        owned_names=EMPIRICAL_OUTPUT_NAMES,
    )
    temporary.cleanup()

    print(
        f"Wrote empirical results for {len(groups)} groups and {len(draw_manifest)} draws.",
        file=sys.stderr,
    )
    return 0


def _probabilities_command(args: argparse.Namespace) -> int:
    probabilities = dgs_probabilities(
        s=args.selfing_rate,
        N=args.N,
        n_diploids=args.n_diploids,
    )
    branch_lengths = expected_dgs_branch_lengths(
        s=args.selfing_rate,
        N=args.N,
        n_diploids=args.n_diploids,
        use_generation_time_scale=args.generation_time_scale,
    )
    rows = [
        {
            "n0": config[0],
            "n1": config[1],
            "n2": config[2],
            "probability": probabilities[config],
            "expected_branch_length": branch_lengths[config],
        }
        for config in sorted(probabilities)
    ]
    frame = pd.DataFrame(
        rows,
        columns=["n0", "n1", "n2", "probability", "expected_branch_length"],
    )
    if args.out:
        frame.to_csv(args.out, index=False)
    else:
        frame.to_csv(sys.stdout, index=False)
    return 0


def _simulate_command(args: argparse.Namespace) -> int:
    config = ValidationExperimentConfig(
        slim_script=args.slim_script,
        outdir=args.out,
        census_size=args.census_size,
        true_s=args.selfing_rate,
        n_loci=args.n_loci,
        sampling_design=args.sampling_design,
        chrom_length_each=args.chrom_length,
        mu=args.mu,
        recomb_rate=args.recomb_rate,
        n_sample=args.n_sample,
        burn_mult=args.burn_mult,
        n_reps=args.n_reps,
        s_grid=_parse_grid(args.grid),
        seed=args.seed,
        slim_executable=args.slim_executable,
        include_monomorphic=not args.exclude_monomorphic,
        vcf_missing=args.missing,
        vcf_malformed=args.malformed,
        vcf_multiallelic=args.multiallelic,
        vcf_require_pass=not args.include_nonpass,
        vcf_polarization=args.polarization,
        vcf_outgroup_sample_ids=_parse_sample_list(args.outgroup_samples),
        vcf_min_outgroup_called=args.min_outgroup_called,
        vcf_require_homozygous_outgroup=not args.allow_heterozygous_outgroup,
        vcf_skip_polymorphic_outgroup=not (
            args.allow_polymorphic_outgroup or args.allow_heterozygous_outgroup
        ),
        fit_error_rate=args.fit_error_rate,
        fit_mode=args.fit_mode,
    )
    run_validation_experiment(config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="selfdgs",
        description="DGS tools for partial-selfing inference.",
    )
    parser.add_argument("--version", action="version", version=f"selfdgs {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dgs_parser = subparsers.add_parser("dgs", help="Convert VCF/BCF to DGS counts.")
    dgs_parser.add_argument("vcf")
    dgs_parser.add_argument("--out", required=True, help="Output DGS CSV path.")
    _add_vcf_options(dgs_parser)
    dgs_parser.set_defaults(func=_dgs_command)

    fit_parser = subparsers.add_parser("fit", help="Estimate selfing rate.")
    fit_parser.add_argument(
        "input", help="Input VCF/BCF path, or DGS CSV with --dgs-csv."
    )
    fit_parser.add_argument("--out", required=True, help="Output directory.")
    fit_parser.add_argument("--dgs-csv", action="store_true", help="Treat input as DGS CSV.")
    fit_parser.add_argument("--grid", default=None, help="Grid as start:stop:num or comma list.")
    fit_parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Numerically floor each cell probability without renormalizing.",
    )
    fit_parser.add_argument(
        "--error-rate",
        type=float,
        default=0.0,
        help=(
            "Mix model probabilities with a normalized uniform distribution "
            "over model and positive-count observed DGS cells."
        ),
    )
    fit_parser.add_argument(
        "--support-drop",
        type=float,
        default=1.92,
        help="Delta log-likelihood cutoff for descriptive, uncalibrated support.",
    )
    fit_parser.add_argument("--refine", action="store_true", help="Refine the grid optimum.")
    fit_parser.add_argument("--refine-lower", type=float, default=0.0)
    fit_parser.add_argument("--refine-upper", type=float, default=1.0)
    fit_parser.add_argument(
        "--mode",
        choices=["unfolded", "folded"],
        default=None,
        help=(
            "Likelihood model mode. Defaults to folded when --polarization "
            "folded is used, otherwise unfolded."
        ),
    )
    _add_vcf_options(fit_parser, allow_skip_genotype=False)
    fit_parser.set_defaults(func=_fit_command)

    empirical_parser = subparsers.add_parser(
        "empirical",
        help="Run a grouped empirical draw workflow from VCF/BCF.",
    )
    empirical_parser.add_argument("vcf")
    empirical_parser.add_argument("--out", required=True, help="Output directory.")
    empirical_parser.add_argument(
        "--n-diploids",
        type=int,
        required=True,
        help="Number of diploid focal samples in each draw.",
    )
    empirical_parser.add_argument(
        "--n-draws",
        type=int,
        default=1,
        help="Number of random draws per sample group.",
    )
    empirical_parser.add_argument("--seed", type=int, default=1, help="Random draw seed.")
    empirical_parser.add_argument(
        "--mode",
        choices=["unfolded", "folded", "aa", "outgroup-consensus"],
        default="folded",
        help=(
            "VCF counting/polarization mode for empirical DGS counts "
            "(default: folded)."
        ),
    )
    empirical_parser.add_argument(
        "--fit-mode",
        choices=["unfolded", "folded"],
        default=None,
        help="Likelihood mode. Defaults to folded only when --mode folded is used.",
    )
    empirical_parser.add_argument(
        "--metadata",
        default=None,
        help="Metadata CSV used for grouped analyses or outgroup selection.",
    )
    empirical_parser.add_argument(
        "--sample-column",
        default="sample",
        help="Metadata column containing VCF sample IDs.",
    )
    empirical_parser.add_argument(
        "--group-column",
        default=None,
        help="Metadata column defining focal sample groups.",
    )
    empirical_parser.add_argument(
        "--grouping",
        choices=["auto", "all", "grouped", "both"],
        default="auto",
        help=(
            "Sample grouping mode. 'auto' uses metadata grouping when metadata "
            "and --group-column are supplied, otherwise all VCF samples."
        ),
    )
    empirical_parser.add_argument(
        "--all-group-name",
        default="all_samples",
        help="Group name used for all-sample grouping.",
    )
    empirical_parser.add_argument(
        "--sample-list",
        action="append",
        default=[],
        metavar="[GROUP=]PATH",
        help=(
            "Explicit focal sample list file. May be repeated. Without GROUP=, "
            "the file stem is used as the group name."
        ),
    )
    empirical_parser.add_argument(
        "--outgroup-column",
        default=None,
        help="Metadata column used to select outgroup samples.",
    )
    empirical_parser.add_argument(
        "--outgroup-value",
        default=None,
        help="Metadata value identifying outgroup samples.",
    )
    empirical_parser.add_argument(
        "--outgroup-samples",
        default=None,
        help="Comma-separated outgroup sample IDs for outgroup-consensus mode.",
    )
    empirical_parser.add_argument(
        "--min-outgroup-called",
        type=int,
        default=1,
        help="Minimum called outgroup genotypes required for outgroup consensus.",
    )
    empirical_parser.add_argument(
        "--allow-heterozygous-outgroup",
        action="store_true",
        help=(
            "Allow heterozygous outgroup genotypes and infer the majority allele "
            "when the outgroup is polymorphic."
        ),
    )
    empirical_parser.add_argument(
        "--allow-polymorphic-outgroup",
        action="store_true",
        help="Allow polymorphic outgroups and use the majority allele as ancestral.",
    )
    empirical_parser.add_argument(
        "--missing",
        choices=["error", "skip-draw", "skip-site"],
        default="skip-draw",
        help="Policy for missing focal genotypes in empirical draws.",
    )
    empirical_parser.add_argument(
        "--malformed",
        choices=["error", "skip-draw", "skip-site"],
        default="error",
        help="Policy for malformed focal genotypes in empirical draws.",
    )
    empirical_parser.add_argument(
        "--multiallelic",
        choices=["skip", "error"],
        default="skip",
        help="Policy for multiallelic VCF records.",
    )
    empirical_parser.add_argument(
        "--include-nonpass",
        action="store_true",
        help="Include VCF records with FILTER values other than PASS or '.'.",
    )
    empirical_parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Stop after this many VCF records; mainly useful for audits/tests.",
    )
    empirical_parser.add_argument("--grid", default=None, help="Grid as start:stop:num or comma list.")
    empirical_parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Numerically floor each cell probability without renormalizing.",
    )
    empirical_parser.add_argument(
        "--error-rate",
        type=float,
        default=0.0,
        help=(
            "Mix model probabilities with a normalized uniform distribution "
            "over model and positive-count observed DGS cells."
        ),
    )
    empirical_parser.add_argument(
        "--support-drop",
        type=float,
        default=1.92,
        help="Delta log-likelihood cutoff for descriptive, uncalibrated draw support.",
    )
    empirical_parser.add_argument(
        "--low-polymorphic-site-threshold",
        type=int,
        default=50,
        help="Warning threshold for low polymorphic-site counts; set negative to disable.",
    )
    empirical_parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write sample_audit.csv and draw_manifest.csv without counting or fitting.",
    )
    empirical_parser.add_argument(
        "--plots",
        action="store_true",
        help="Also write empirical PNG plots.",
    )
    empirical_parser.set_defaults(func=_empirical_command)

    prob_parser = subparsers.add_parser(
        "probabilities",
        help="Write expected DGS probabilities for a selfing rate.",
    )
    prob_parser.add_argument("--selfing-rate", "-s", type=float, required=True)
    prob_parser.add_argument("--n-diploids", type=int, default=4)
    prob_parser.add_argument(
        "--N",
        type=float,
        default=1.0,
        help=(
            "Effective-size scale used only for expected branch lengths when "
            "--generation-time-scale is supplied."
        ),
    )
    prob_parser.add_argument(
        "--generation-time-scale",
        action="store_true",
        help="Report expected branch lengths scaled in generations.",
    )
    prob_parser.add_argument("--out", default=None, help="Output CSV path; defaults to stdout.")
    prob_parser.set_defaults(func=_probabilities_command)

    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Run a SLiM validation experiment.",
    )
    simulate_parser.add_argument(
        "--slim-script",
        default=None,
        help="Custom SLiM script implementing the selected sampling-design contract.",
    )
    simulate_parser.add_argument("--out", required=True, help="Output directory.")
    simulate_parser.add_argument("--slim-executable", default="slim")
    simulate_parser.add_argument("--census-size", "--ne", dest="census_size", type=int, default=500, help="Diploid census population size.")
    simulate_parser.add_argument(
        "--sampling-design", choices=["fixed_individuals", "independent_populations"],
        default="fixed_individuals",
        help="Fixed individuals across shared-pedigree loci, or a separate population per locus.",
    )
    simulate_parser.add_argument("--selfing-rate", type=float, default=0.5)
    simulate_parser.add_argument("--n-reps", type=int, default=1)
    simulate_parser.add_argument("--n-loci", type=int, default=1, help="Number of loci per fitted replicate.")
    simulate_parser.add_argument("--chrom-length", type=int, default=10_000)
    simulate_parser.add_argument("--mu", type=float, default=1e-7)
    simulate_parser.add_argument("--recomb-rate", type=float, default=5e-8)
    simulate_parser.add_argument(
        "--n-sample",
        "--n-diploids",
        dest="n_sample",
        type=int,
        default=4,
        help="Number of sampled diploid individuals per fitted dataset.",
    )
    simulate_parser.add_argument("--burn-mult", type=int, default=10)
    simulate_parser.add_argument("--seed", type=int, default=12345)
    simulate_parser.add_argument("--grid", default=None, help="Grid as start:stop:num or comma list.")
    simulate_parser.add_argument(
        "--fit-error-rate",
        type=float,
        default=0.0,
        help=(
            "Mix fit probabilities with a normalized uniform distribution "
            "over model and positive-count observed DGS cells."
        ),
    )
    simulate_parser.add_argument(
        "--fit-mode",
        choices=["unfolded", "folded"],
        default=None,
        help=(
            "Likelihood model mode. Defaults to folded when --polarization "
            "folded is used, otherwise unfolded."
        ),
    )
    _add_vcf_options(simulate_parser, include_n_diploids=False, allow_skip_genotype=False)
    simulate_parser.set_defaults(func=_simulate_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"selfdgs: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
