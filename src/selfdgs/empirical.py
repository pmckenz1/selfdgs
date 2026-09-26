"""Empirical grouped-draw DGS workflows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from urllib.parse import quote

import numpy as np
import pandas as pd

from selfdgs._outputs import install_staged_outputs
from selfdgs.fit import (
    AllNonfiniteLikelihoodError,
    _regularization_name,
    fit_selfing,
)
from selfdgs.io.vcf import (
    Polarization,
    read_vcf_samples,
    scan_vcf_dgs,
)
from selfdgs.likelihood import (
    LikelihoodMode,
    _validate_regularization,
    model_probabilities_for_mode,
)
from selfdgs.spectrum import (
    DGSConfig,
    DGSCounts,
    filter_polymorphic_dgs,
    write_dgs_csv,
)

DrawKey = tuple[str, int]
DrawMissingPolicy = Literal["error", "skip-draw", "skip-site", "skip-genotype"]
DrawMalformedPolicy = Literal["error", "skip-draw", "skip-site", "skip-genotype"]
EmpiricalMissingPolicy = Literal["error", "skip-draw", "skip-site"]
EmpiricalMalformedPolicy = Literal["error", "skip-draw", "skip-site"]


@dataclass(frozen=True)
class EmpiricalAnalysisConfig:
    """Settings shared by grouped empirical DGS analyses."""

    vcf_path: str | Path
    n_diploids: int
    n_draws: int = 1
    seed: int = 1
    mode: str = "folded"
    fit_mode: LikelihoodMode | None = None
    outgroup_sample_ids: Sequence[str] = field(default_factory=tuple)
    min_outgroup_called: int = 1
    require_homozygous: bool = True
    skip_polymorphic_outgroup: bool = True
    require_pass: bool = True
    max_records: int | None = None
    missing: EmpiricalMissingPolicy = "skip-draw"
    malformed: EmpiricalMalformedPolicy = "error"
    multiallelic: Literal["skip", "error"] = "skip"
    s_grid: Iterable[float] | None = None
    epsilon: float = 0.0
    error_rate: float = 0.0
    support_drop: float = 1.92
    low_polymorphic_site_threshold: int | None = 50
    outdir: str | Path | None = None


@dataclass(frozen=True)
class EmpiricalAnalysisResult:
    """Tables and DGS counts produced by :func:`run_empirical_analysis`."""

    sample_audit: pd.DataFrame
    draw_manifest: pd.DataFrame
    counts_by_draw: dict[DrawKey, DGSCounts]
    fit_counts_by_draw: dict[DrawKey, DGSCounts]
    site_diagnostics: pd.DataFrame
    draw_diagnostics: pd.DataFrame
    fit_summary: pd.DataFrame
    likelihood_curves: pd.DataFrame
    aggregate_summary: pd.DataFrame
    representative_observed_vs_fitted: pd.DataFrame

SAMPLE_AUDIT_COLUMNS = [
    "group",
    "n_samples",
    "metadata_n",
    "in_vcf_n",
    "missing_from_vcf_n",
    "included",
]
DRAW_MANIFEST_COLUMNS = ["group", "draw_id", "n_diploids", "seed", "sample_ids"]
SITE_DIAGNOSTICS_COLUMNS = ["metric", "value"]
DRAW_DIAGNOSTICS_COLUMNS = [
    "group",
    "draw_id",
    "retained_sites",
    "polymorphic_sites",
    "monomorphic_sites",
    "missing_genotype",
    "malformed_genotype",
]
DRAW_FIT_SUMMARY_COLUMNS = [
    "group",
    "draw_id",
    "best_s",
    "best_loglik",
    "n_sites",
    "n_fit_sites",
    "support_lower",
    "support_upper",
    "boundary_optimum",
    "warnings",
    "mode",
    "input_polarization",
    "probability_regularization",
    "epsilon",
    "error_rate",
    "unsupported_cell_count",
    "unsupported_site_count",
]
DRAW_LIKELIHOOD_COLUMNS = ["group", "draw_id", "s", "loglik", "delta_loglik"]
GROUP_FIT_SUMMARY_COLUMNS = [
    "group",
    "n_fit_draws",
    "median_best_s",
    "mean_best_s",
    "q025_best_s",
    "q975_best_s",
    "median_n_fit_sites",
]
REPRESENTATIVE_OBSERVED_VS_FITTED_COLUMNS = [
    "group",
    "representative_draw_id",
    "n0",
    "n1",
    "n2",
    "observed_count",
    "fitted_expected_count",
    "fitted_probability",
    "best_s",
    "n_fit_sites",
    "mode",
    "input_polarization",
    "representative_reason",
]

EMPIRICAL_OUTPUT_NAMES = frozenset(
    {
        "draw_manifest.csv",
        "sample_audit.csv",
        "site_diagnostics.csv",
        "draw_diagnostics.csv",
        "draw_fit_summary.csv",
        "draw_likelihood_curves.csv",
        "group_fit_summary.csv",
        "representative_draw_observed_vs_fitted.csv",
        "grouped_likelihood_summary.png",
        "best_s_distribution.png",
        "representative_observed_vs_fitted.png",
        "dgs",
    }
)


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    extra = [column for column in normalized.columns if column not in columns]
    return normalized[[*columns, *extra]]


def _append_warning(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def _warning_text(warnings: Sequence[str]) -> str:
    return ";".join(warning for warning in warnings if warning)


def _load_metadata(metadata: pd.DataFrame | str | Path | None) -> pd.DataFrame | None:
    if metadata is None:
        return None
    if isinstance(metadata, pd.DataFrame):
        frame = metadata.copy()
    else:
        frame = pd.read_csv(metadata, dtype=str)
    for column in frame.columns:
        frame[column] = frame[column].astype("string").str.strip()
    return frame


def make_sample_groups(
    sample_ids: Iterable[str],
    *,
    mode: Literal["all", "grouped", "both"] = "all",
    metadata: pd.DataFrame | str | Path | None = None,
    sample_column: str = "sample",
    group_column: str | None = None,
    min_samples: int = 1,
    all_group_name: str = "all_samples",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build empirical sample groups from VCF samples and optional metadata."""
    if mode not in {"all", "grouped", "both"}:
        raise ValueError("mode must be 'all', 'grouped', or 'both'.")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1.")

    samples = [str(sample) for sample in sample_ids]
    duplicate_samples = sorted(
        sample for sample, count in Counter(samples).items() if count > 1
    )
    if duplicate_samples:
        raise ValueError(f"VCF sample IDs must be unique; duplicates: {duplicate_samples}")
    sample_set = set(samples)
    groups: dict[str, list[str]] = {}
    rows: list[dict[str, object]] = []

    if mode in {"all", "both"}:
        included = len(samples) >= min_samples
        rows.append(
            {
                "group": all_group_name,
                "n_samples": len(samples),
                "metadata_n": len(samples),
                "in_vcf_n": len(samples),
                "missing_from_vcf_n": 0,
                "included": included,
            }
        )
        if included:
            groups[all_group_name] = samples

    if mode in {"grouped", "both"}:
        frame = _load_metadata(metadata)
        if frame is None:
            raise ValueError("metadata is required when mode is 'grouped' or 'both'.")
        if group_column is None:
            raise ValueError("group_column is required when mode is 'grouped' or 'both'.")
        missing_columns = {sample_column, group_column}.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")
        missing_values = {
            column: int((frame[column].isna() | frame[column].eq("")).sum())
            for column in (sample_column, group_column)
            if (frame[column].isna() | frame[column].eq("")).any()
        }
        if missing_values:
            raise ValueError(
                "Metadata sample/group columns contain missing values: "
                f"{missing_values}"
            )
        duplicate_metadata_samples = sorted(
            frame.loc[frame[sample_column].duplicated(keep=False), sample_column]
            .astype(str)
            .unique()
            .tolist()
        )
        if duplicate_metadata_samples:
            raise ValueError(
                "Metadata sample IDs must be unique; duplicates: "
                f"{duplicate_metadata_samples}"
            )
        if mode == "both" and all_group_name in set(frame[group_column].astype(str)):
            raise ValueError(
                f"Metadata group {all_group_name!r} collides with the all-samples group."
            )

        metadata_by_group = {
            str(group): group_frame[sample_column].astype(str).tolist()
            for group, group_frame in frame.groupby(group_column, sort=True)
        }
        for group, metadata_samples in metadata_by_group.items():
            metadata_set = set(metadata_samples)
            in_vcf = [sample for sample in samples if sample in metadata_set]
            missing_from_vcf = sorted(metadata_set.difference(sample_set))
            included = len(in_vcf) >= min_samples
            rows.append(
                {
                    "group": group,
                    "n_samples": len(in_vcf),
                    "metadata_n": len(metadata_samples),
                    "in_vcf_n": len(in_vcf),
                    "missing_from_vcf_n": len(missing_from_vcf),
                    "included": included,
                }
            )
            if included:
                groups[group] = in_vcf

    return _ensure_columns(pd.DataFrame(rows), SAMPLE_AUDIT_COLUMNS), groups


def make_draw_manifest(
    sample_groups: Mapping[str, Sequence[str]],
    *,
    n_diploids: int,
    n_draws: int,
    seed: int,
) -> pd.DataFrame:
    """Randomly draw fixed-size diploid sample sets from each group.

    Each row receives a deterministic seed derived from the manifest seed,
    group, and draw ID, so it can be regenerated independently.
    """
    if n_diploids < 1:
        raise ValueError("n_diploids must be at least 1.")
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1.")

    rows: list[dict[str, object]] = []
    for group in sorted(sample_groups):
        samples = [str(sample) for sample in sample_groups[group]]
        if len(samples) != len(set(samples)):
            raise ValueError(f"Group {group!r} contains duplicate sample IDs.")
        if len(samples) < n_diploids:
            raise ValueError(f"Group {group!r} has only {len(samples)} samples; need {n_diploids}.")
        for draw_id in range(n_draws):
            digest = hashlib.sha256(
                f"{seed}\0{group}\0{draw_id}".encode("utf-8")
            ).digest()
            draw_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
            rng = np.random.default_rng(draw_seed)
            selected = rng.choice(samples, size=n_diploids, replace=False).tolist()
            rows.append(
                {
                    "group": group,
                    "draw_id": draw_id,
                    "n_diploids": n_diploids,
                    "seed": draw_seed,
                    "sample_ids": ";".join(selected),
                }
            )
    return pd.DataFrame(rows, columns=DRAW_MANIFEST_COLUMNS)


def _validate_draw_manifest(draw_manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"group", "draw_id", "sample_ids"}
    missing = required.difference(draw_manifest.columns)
    if missing:
        raise ValueError(f"draw_manifest is missing required columns: {sorted(missing)}")
    draws = draw_manifest.copy()
    draws["group"] = draws["group"].astype(str)
    draws["draw_id"] = draws["draw_id"].astype(int)
    duplicate_keys = draws.duplicated(["group", "draw_id"], keep=False)
    if duplicate_keys.any():
        duplicates = (
            draws.loc[duplicate_keys, ["group", "draw_id"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(
            "draw_manifest contains duplicate (group, draw_id) keys: "
            f"{duplicates[:10]}"
        )
    draws["sample_list"] = draws["sample_ids"].astype(str).str.split(";")
    return draws


def _normalize_draw_polarization(*, mode: str, polarization: str | None) -> Polarization:
    strategy = polarization or mode
    if strategy == "unfolded":
        strategy = "ref"
    if strategy not in {"ref", "folded", "aa", "outgroup-consensus"}:
        raise ValueError(
            "mode/polarization must be 'unfolded', 'ref', 'folded', 'aa', "
            "or 'outgroup-consensus'."
        )
    return strategy  # type: ignore[return-value]


def dgs_for_draws(
    vcf_path: str | Path,
    draw_manifest: pd.DataFrame,
    *,
    mode: str = "folded",
    polarization: Polarization | None = None,
    outgroup_sample_ids: Sequence[str] | None = None,
    min_outgroup_called: int = 1,
    require_homozygous: bool = True,
    skip_polymorphic_outgroup: bool = True,
    require_pass: bool = True,
    max_records: int | None = None,
    missing: DrawMissingPolicy = "skip-draw",
    malformed: DrawMalformedPolicy = "skip-draw",
    multiallelic: Literal["skip", "error"] = "skip",
    include_monomorphic: bool = True,
) -> tuple[dict[DrawKey, DGSCounts], pd.DataFrame, pd.DataFrame]:
    """Count DGS cells for all manifest draws using the shared VCF scanner."""
    polarization_strategy = _normalize_draw_polarization(mode=mode, polarization=polarization)
    draws = _validate_draw_manifest(draw_manifest)
    sample_sets: dict[DrawKey, list[str]] = {
        (row.group, int(row.draw_id)): list(row.sample_list)
        for row in draws.itertuples(index=False)
    }
    scan = scan_vcf_dgs(
        vcf_path,
        sample_sets=sample_sets,
        include_monomorphic=include_monomorphic,
        missing=missing,
        malformed=malformed,
        multiallelic=multiallelic,
        polarization=polarization_strategy,
        outgroup_sample_ids=outgroup_sample_ids,
        min_outgroup_called=min_outgroup_called,
        require_homozygous=require_homozygous,
        skip_polymorphic_outgroup=skip_polymorphic_outgroup,
        require_pass=require_pass,
        max_records=max_records,
    )

    site_diagnostics_df = _ensure_columns(
        pd.DataFrame(
            [
                {"metric": metric, "value": value}
                for metric, value in sorted(scan.site_diagnostics.items())
            ]
        ),
        SITE_DIAGNOSTICS_COLUMNS,
    )
    draw_diagnostics_df = _ensure_columns(
        pd.DataFrame(
            [
                {
                    "group": key[0],
                    "draw_id": key[1],
                    **{
                        metric: diagnostics.get(metric, 0)
                        for metric in DRAW_DIAGNOSTICS_COLUMNS[2:]
                    },
                }
                for key, diagnostics in sorted(scan.set_diagnostics.items())
            ]
        ),
        DRAW_DIAGNOSTICS_COLUMNS,
    )

    return scan.counts_by_set, site_diagnostics_df, draw_diagnostics_df  # type: ignore[return-value]


def _fit_warnings(
    *,
    existing: Sequence[str] = (),
    n_polymorphic: int,
    low_polymorphic_site_threshold: int | None,
    likelihood_curve=(),
    support_interval: tuple[float, float] | None = None,
) -> str:
    warnings = list(existing)
    if (
        low_polymorphic_site_threshold is not None
        and n_polymorphic < low_polymorphic_site_threshold
    ):
        _append_warning(
            warnings,
            f"low polymorphic site count (<{low_polymorphic_site_threshold})",
        )
    logliks = [float(point.loglik) for point in likelihood_curve]
    if logliks and not any(math.isfinite(loglik) for loglik in logliks):
        _append_warning(warnings, "all-nonfinite likelihoods")
    if support_interval is not None and likelihood_curve:
        grid_values = [float(point.s) for point in likelihood_curve]
        support_lower, support_upper = support_interval
        if support_lower <= min(grid_values) or support_upper >= max(grid_values):
            _append_warning(warnings, "support interval touches grid boundary")
    return _warning_text(warnings)


def fit_result_to_draw_tables(
    fit,
    counts: Mapping[DGSConfig, int],
    *,
    group: str,
    draw_id: int,
    mode: LikelihoodMode,
    input_polarization: str,
    low_polymorphic_site_threshold: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Represent one successful fit in the shared draw-level table schemas."""
    polymorphic = filter_polymorphic_dgs(counts, n_diploids=fit.n_diploids)
    n_sites = int(sum(counts.values()))
    n_polymorphic = int(sum(polymorphic.values()))
    support_lower, support_upper = (np.nan, np.nan)
    if fit.support_interval is not None:
        support_lower, support_upper = fit.support_interval
    warning_text = _fit_warnings(
        existing=fit.warnings,
        n_polymorphic=n_polymorphic,
        low_polymorphic_site_threshold=low_polymorphic_site_threshold,
        likelihood_curve=fit.likelihood_curve,
        support_interval=fit.support_interval,
    )
    summary = _ensure_columns(
        pd.DataFrame(
            [
                {
                    "group": group,
                    "draw_id": int(draw_id),
                    "best_s": fit.best_s,
                    "best_loglik": fit.best_loglik,
                    "n_sites": n_sites,
                    "n_fit_sites": n_polymorphic,
                    "support_lower": support_lower,
                    "support_upper": support_upper,
                    "boundary_optimum": fit.boundary_optimum,
                    "warnings": warning_text,
                    "mode": mode,
                    "input_polarization": input_polarization,
                    "probability_regularization": fit.metadata[
                        "probability_regularization"
                    ],
                    "epsilon": fit.metadata["epsilon"],
                    "error_rate": fit.metadata["error_rate"],
                    "unsupported_cell_count": fit.metadata["unsupported_cell_count"],
                    "unsupported_site_count": fit.metadata["unsupported_site_count"],
                }
            ]
        ),
        DRAW_FIT_SUMMARY_COLUMNS,
    )
    likelihood = fit.likelihood_dataframe()
    likelihood.insert(0, "draw_id", int(draw_id))
    likelihood.insert(0, "group", group)
    return summary, _ensure_columns(likelihood, DRAW_LIKELIHOOD_COLUMNS)


def fit_draws(
    counts_by_draw: Mapping[DrawKey, DGSCounts],
    *,
    n_diploids: int,
    mode: LikelihoodMode = "unfolded",
    s_grid: Iterable[float] | None = None,
    N: float = 1.0,
    epsilon: float = 0.0,
    error_rate: float = 0.0,
    support_drop: float = 1.92,
    input_polarization: str | None = None,
    low_polymorphic_site_threshold: int | None = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[DrawKey, DGSCounts]]:
    """Fit selfing-rate likelihoods for empirical draw DGS counts."""
    if mode not in {"unfolded", "folded"}:
        raise ValueError("mode must be 'unfolded' or 'folded'.")
    _validate_regularization(epsilon, error_rate)
    if not math.isfinite(support_drop) or support_drop < 0:
        raise ValueError("support_drop must be finite and non-negative.")
    if low_polymorphic_site_threshold is not None and low_polymorphic_site_threshold < 0:
        raise ValueError("low_polymorphic_site_threshold must be non-negative or None.")
    grid = None if s_grid is None else [float(s) for s in s_grid]
    recorded_polarization = input_polarization or ("folded" if mode == "folded" else "ref")
    fit_counts_by_draw: dict[DrawKey, DGSCounts] = {}
    summary_rows: list[dict[str, object]] = []
    likelihood_frames: list[pd.DataFrame] = []

    for key, counts in sorted(counts_by_draw.items()):
        group, draw_id = key
        polymorphic = filter_polymorphic_dgs(counts, n_diploids=n_diploids)
        fit_counts = polymorphic
        fit_counts_by_draw[key] = fit_counts
        n_sites = int(sum(counts.values()))
        n_polymorphic = int(sum(polymorphic.values()))
        n_fit_sites = int(sum(fit_counts.values()))

        if n_fit_sites == 0:
            warnings = ["no fit sites"]
            warning_text = _fit_warnings(
                existing=warnings,
                n_polymorphic=n_polymorphic,
                low_polymorphic_site_threshold=low_polymorphic_site_threshold,
            )
            summary_rows.append(
                {
                    "group": group,
                    "draw_id": draw_id,
                    "best_s": np.nan,
                    "best_loglik": np.nan,
                    "n_sites": n_sites,
                    "n_fit_sites": 0,
                    "support_lower": np.nan,
                    "support_upper": np.nan,
                    "boundary_optimum": False,
                    "warnings": warning_text,
                    "mode": mode,
                    "input_polarization": recorded_polarization,
                    "probability_regularization": _regularization_name(
                        epsilon=epsilon,
                        error_rate=error_rate,
                    ),
                    "epsilon": epsilon,
                    "error_rate": error_rate,
                    "unsupported_cell_count": 0,
                    "unsupported_site_count": 0,
                }
            )
            continue

        try:
            fit = fit_selfing(
                counts,
                N=N,
                n_diploids=n_diploids,
                s_grid=grid,
                epsilon=epsilon,
                error_rate=error_rate,
                support_drop=support_drop,
                mode=mode,
                input_polarization=recorded_polarization,
            )
        except AllNonfiniteLikelihoodError as exc:
            unsupported_sites = int(sum(exc.unsupported_cells.values()))
            unsupported_text = ",".join(
                f"{config}:{count}"
                for config, count in sorted(exc.unsupported_cells.items())
            )
            warning_text = _fit_warnings(
                existing=(
                    "all-nonfinite likelihoods",
                    f"unsupported DGS cells ({unsupported_text})",
                ),
                n_polymorphic=n_polymorphic,
                low_polymorphic_site_threshold=low_polymorphic_site_threshold,
            )
            summary_rows.append(
                {
                    "group": group,
                    "draw_id": draw_id,
                    "best_s": np.nan,
                    "best_loglik": np.nan,
                    "n_sites": n_sites,
                    "n_fit_sites": n_fit_sites,
                    "support_lower": np.nan,
                    "support_upper": np.nan,
                    "boundary_optimum": False,
                    "warnings": warning_text,
                    "mode": mode,
                    "input_polarization": recorded_polarization,
                    "probability_regularization": _regularization_name(
                        epsilon=epsilon,
                        error_rate=error_rate,
                    ),
                    "epsilon": epsilon,
                    "error_rate": error_rate,
                    "unsupported_cell_count": len(exc.unsupported_cells),
                    "unsupported_site_count": unsupported_sites,
                }
            )
            continue
        summary, likelihood = fit_result_to_draw_tables(
            fit,
            counts,
            group=group,
            draw_id=draw_id,
            mode=mode,
            input_polarization=recorded_polarization,
            low_polymorphic_site_threshold=low_polymorphic_site_threshold,
        )
        summary_rows.extend(summary.to_dict(orient="records"))
        likelihood_frames.append(likelihood)

    fit_summary = _ensure_columns(pd.DataFrame(summary_rows), DRAW_FIT_SUMMARY_COLUMNS)
    likelihood_curves = (
        _ensure_columns(pd.concat(likelihood_frames, ignore_index=True), DRAW_LIKELIHOOD_COLUMNS)
        if likelihood_frames
        else pd.DataFrame(columns=DRAW_LIKELIHOOD_COLUMNS)
    )
    return fit_summary, likelihood_curves, fit_counts_by_draw


def aggregate_fit_summary(fit_summary: pd.DataFrame, *, group_column: str = "group") -> pd.DataFrame:
    """Summarize draw-level fit results by group."""
    if fit_summary.empty:
        return pd.DataFrame(columns=GROUP_FIT_SUMMARY_COLUMNS)
    if group_column not in fit_summary.columns:
        raise ValueError(f"fit_summary is missing group column {group_column!r}.")

    groups = fit_summary[[group_column]].drop_duplicates()
    fitted_summary = (
        fit_summary.dropna(subset=["best_s"])
        .groupby(group_column, as_index=False)
        .agg(
            n_fit_draws=("best_s", "size"),
            median_best_s=("best_s", "median"),
            mean_best_s=("best_s", "mean"),
            q025_best_s=("best_s", lambda x: x.quantile(0.025)),
            q975_best_s=("best_s", lambda x: x.quantile(0.975)),
            median_n_fit_sites=("n_fit_sites", "median"),
        )
    )
    summary = groups.merge(fitted_summary, on=group_column, how="left")
    summary["n_fit_draws"] = summary["n_fit_draws"].fillna(0).astype(int)
    return _ensure_columns(summary, GROUP_FIT_SUMMARY_COLUMNS)


def representative_observed_vs_fitted(
    fit_summary: pd.DataFrame,
    fit_counts_by_draw: Mapping[DrawKey, DGSCounts],
    *,
    n_diploids: int,
    N: float = 1.0,
    group_column: str = "group",
) -> pd.DataFrame:
    """Create observed-vs-fitted DGS rows for one representative draw per group."""
    if fit_summary.empty:
        return pd.DataFrame(columns=REPRESENTATIVE_OBSERVED_VS_FITTED_COLUMNS)
    if group_column not in fit_summary.columns:
        raise ValueError(f"fit_summary is missing group column {group_column!r}.")

    rows: list[dict[str, object]] = []
    fitted = fit_summary.dropna(subset=["best_s"]).copy()
    if "n_fit_sites" not in fitted.columns:
        fitted["n_fit_sites"] = 0
    fitted = fitted[fitted["n_fit_sites"].astype(float) > 0]
    for group, group_frame in fitted.groupby(group_column, sort=True):
        median_best_s = group_frame["best_s"].median()
        representatives = group_frame.assign(
            _distance=(group_frame["best_s"] - median_best_s).abs()
        ).sort_values(["_distance", "draw_id"])
        representative = representatives.iloc[0]
        draw_id = int(representative["draw_id"])
        key = (str(group), draw_id)
        counts = Counter(fit_counts_by_draw.get(key, Counter()))
        n_fit_sites = int(sum(counts.values()))
        if n_fit_sites <= 0:
            continue

        mode = str(representative.get("mode", "unfolded"))
        probabilities = model_probabilities_for_mode(
            s=float(representative["best_s"]),
            N=N,
            n_diploids=n_diploids,
            mode=mode,  # type: ignore[arg-type]
        )
        epsilon = float(representative.get("epsilon", 0.0))
        error_rate = float(representative.get("error_rate", 0.0))
        support = set(counts).union(probabilities)
        background = 1.0 / len(support) if error_rate > 0.0 else 0.0
        for config in sorted(support):
            probability = float(probabilities.get(config, 0.0))
            if error_rate > 0.0:
                probability = (1.0 - error_rate) * probability + error_rate * background
            if epsilon > 0.0:
                probability = max(probability, epsilon)
            rows.append(
                {
                    "group": str(group),
                    "representative_draw_id": draw_id,
                    "n0": config[0],
                    "n1": config[1],
                    "n2": config[2],
                    "observed_count": int(counts.get(config, 0)),
                    "fitted_expected_count": probability * n_fit_sites,
                    "fitted_probability": probability,
                    "best_s": float(representative["best_s"]),
                    "n_fit_sites": n_fit_sites,
                    "mode": mode,
                    "input_polarization": str(representative.get("input_polarization", "")),
                    "representative_reason": "closest_to_group_median_best_s",
                }
            )
    return _ensure_columns(
        pd.DataFrame(rows),
        REPRESENTATIVE_OBSERVED_VS_FITTED_COLUMNS,
    )


def _safe_label(value: str) -> str:
    """Return a reversible, collision-free label suitable for one path component."""
    return quote(value, safe="")


def _write_draw_outputs_to_directory(
    outdir: Path,
    *,
    counts_by_draw: Mapping[DrawKey, DGSCounts],
    fit_counts_by_draw: Mapping[DrawKey, DGSCounts],
    draw_manifest: pd.DataFrame,
    site_diagnostics: pd.DataFrame,
    draw_diagnostics: pd.DataFrame,
    fit_summary: pd.DataFrame,
    likelihood_curves: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
    representative_observed_vs_fitted: pd.DataFrame | None = None,
    sample_audit: pd.DataFrame | None = None,
) -> None:
    """Write one complete set of empirical outputs to an empty directory."""
    dgs_dir = outdir / "dgs"
    dgs_dir.mkdir()

    _ensure_columns(draw_manifest, DRAW_MANIFEST_COLUMNS).to_csv(
        outdir / "draw_manifest.csv",
        index=False,
    )
    if sample_audit is not None:
        _ensure_columns(sample_audit, SAMPLE_AUDIT_COLUMNS).to_csv(
            outdir / "sample_audit.csv",
            index=False,
        )
    _ensure_columns(site_diagnostics, SITE_DIAGNOSTICS_COLUMNS).to_csv(
        outdir / "site_diagnostics.csv",
        index=False,
    )
    _ensure_columns(draw_diagnostics, DRAW_DIAGNOSTICS_COLUMNS).to_csv(
        outdir / "draw_diagnostics.csv",
        index=False,
    )
    _ensure_columns(fit_summary, DRAW_FIT_SUMMARY_COLUMNS).to_csv(
        outdir / "draw_fit_summary.csv",
        index=False,
    )
    _ensure_columns(likelihood_curves, DRAW_LIKELIHOOD_COLUMNS).to_csv(
        outdir / "draw_likelihood_curves.csv",
        index=False,
    )
    _ensure_columns(aggregate_summary, GROUP_FIT_SUMMARY_COLUMNS).to_csv(
        outdir / "group_fit_summary.csv",
        index=False,
    )
    representative = (
        pd.DataFrame(columns=REPRESENTATIVE_OBSERVED_VS_FITTED_COLUMNS)
        if representative_observed_vs_fitted is None
        else representative_observed_vs_fitted
    )
    _ensure_columns(representative, REPRESENTATIVE_OBSERVED_VS_FITTED_COLUMNS).to_csv(
        outdir / "representative_draw_observed_vs_fitted.csv",
        index=False,
    )

    for group, draw_id in sorted(counts_by_draw):
        prefix = f"{_safe_label(group)}_draw_{draw_id:03d}"
        key = (group, draw_id)
        write_dgs_csv(counts_by_draw[key], dgs_dir / f"{prefix}_dgs.csv")
        write_dgs_csv(fit_counts_by_draw.get(key, Counter()), dgs_dir / f"{prefix}_fit_dgs.csv")


def write_draw_outputs(
    outdir: str | Path,
    *,
    counts_by_draw: Mapping[DrawKey, DGSCounts],
    fit_counts_by_draw: Mapping[DrawKey, DGSCounts],
    draw_manifest: pd.DataFrame,
    site_diagnostics: pd.DataFrame,
    draw_diagnostics: pd.DataFrame,
    fit_summary: pd.DataFrame,
    likelihood_curves: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
    representative_observed_vs_fitted: pd.DataFrame | None = None,
    sample_audit: pd.DataFrame | None = None,
) -> None:
    """Transactionally replace empirical outputs while preserving other files."""
    outdir = Path(outdir)
    outdir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{outdir.name}.", dir=outdir.parent) as tempdir:
        staged_outdir = Path(tempdir) / "output"
        staged_outdir.mkdir()
        _write_draw_outputs_to_directory(
            staged_outdir,
            counts_by_draw=counts_by_draw,
            fit_counts_by_draw=fit_counts_by_draw,
            draw_manifest=draw_manifest,
            site_diagnostics=site_diagnostics,
            draw_diagnostics=draw_diagnostics,
            fit_summary=fit_summary,
            likelihood_curves=likelihood_curves,
            aggregate_summary=aggregate_summary,
            representative_observed_vs_fitted=representative_observed_vs_fitted,
            sample_audit=sample_audit,
        )
        install_staged_outputs(
            staged_outdir,
            outdir,
            owned_names=EMPIRICAL_OUTPUT_NAMES,
        )


def run_empirical_analysis(
    config: EmpiricalAnalysisConfig,
    *,
    sample_groups: Mapping[str, Sequence[str]] | None = None,
    sample_audit: pd.DataFrame | None = None,
) -> EmpiricalAnalysisResult:
    """Run the standard empirical workflow and optionally write its outputs.

    Dataset-specific code may supply biologically meaningful ``sample_groups``.
    With no groups, all VCF samples are analyzed as one group.
    """
    for policy_name, policy in (
        ("missing", config.missing),
        ("malformed", config.malformed),
    ):
        if policy not in {"error", "skip-draw", "skip-site"}:
            raise ValueError(
                f"Empirical {policy_name} policy must preserve the fixed draw size; "
                f"got {policy!r}."
            )
    vcf_samples = read_vcf_samples(config.vcf_path)
    outgroup_set = (
        set(config.outgroup_sample_ids)
        if config.mode == "outgroup-consensus"
        else set()
    )
    if sample_groups is None:
        focal_samples = [sample for sample in vcf_samples if sample not in outgroup_set]
        generated_audit, groups = make_sample_groups(
            focal_samples,
            mode="all",
            min_samples=config.n_diploids,
        )
        audit = generated_audit if sample_audit is None else sample_audit.copy()
    else:
        vcf_sample_set = set(vcf_samples)
        groups = {str(group): [str(sample) for sample in samples] for group, samples in sample_groups.items()}
        missing = {
            group: [sample for sample in samples if sample not in vcf_sample_set]
            for group, samples in groups.items()
        }
        missing = {group: samples for group, samples in missing.items() if samples}
        if missing:
            raise ValueError(f"Sample groups contain samples absent from VCF: {missing}")
        overlapping = {
            group: sorted(set(samples).intersection(outgroup_set))
            for group, samples in groups.items()
        }
        overlapping = {
            group: samples for group, samples in overlapping.items() if samples
        }
        if overlapping:
            raise ValueError(
                "Sample groups contain outgroup samples that cannot also be focal: "
                f"{overlapping}"
            )
        too_small = {group: len(samples) for group, samples in groups.items() if len(samples) < config.n_diploids}
        if too_small:
            raise ValueError(f"Sample groups have fewer than {config.n_diploids} samples: {too_small}")
        if sample_audit is None:
            audit = _ensure_columns(
                pd.DataFrame(
                    {
                        "group": group,
                        "n_samples": len(samples),
                        "metadata_n": len(samples),
                        "in_vcf_n": len(samples),
                        "missing_from_vcf_n": 0,
                        "included": True,
                    }
                    for group, samples in sorted(groups.items())
                ),
                SAMPLE_AUDIT_COLUMNS,
            )
        else:
            audit = _ensure_columns(sample_audit, SAMPLE_AUDIT_COLUMNS)
    if not groups:
        raise ValueError("No sample groups were supplied for empirical analysis.")

    draw_manifest = make_draw_manifest(
        groups,
        n_diploids=config.n_diploids,
        n_draws=config.n_draws,
        seed=config.seed,
    )
    counts, site_diagnostics, draw_diagnostics = dgs_for_draws(
        config.vcf_path,
        draw_manifest,
        mode=config.mode,
        outgroup_sample_ids=config.outgroup_sample_ids or None,
        min_outgroup_called=config.min_outgroup_called,
        require_homozygous=config.require_homozygous,
        skip_polymorphic_outgroup=config.skip_polymorphic_outgroup,
        require_pass=config.require_pass,
        max_records=config.max_records,
        missing=config.missing,
        malformed=config.malformed,
        multiallelic=config.multiallelic,
    )
    fit_mode = config.fit_mode or ("folded" if config.mode == "folded" else "unfolded")
    fit_summary, likelihood_curves, fit_counts = fit_draws(
        counts,
        n_diploids=config.n_diploids,
        mode=fit_mode,
        s_grid=config.s_grid,
        epsilon=config.epsilon,
        error_rate=config.error_rate,
        support_drop=config.support_drop,
        input_polarization="ref" if config.mode == "unfolded" else config.mode,
        low_polymorphic_site_threshold=config.low_polymorphic_site_threshold,
    )
    aggregate = aggregate_fit_summary(fit_summary)
    representative = representative_observed_vs_fitted(
        fit_summary,
        fit_counts,
        n_diploids=config.n_diploids,
    )
    result = EmpiricalAnalysisResult(
        sample_audit=audit,
        draw_manifest=draw_manifest,
        counts_by_draw=counts,
        fit_counts_by_draw=fit_counts,
        site_diagnostics=site_diagnostics,
        draw_diagnostics=draw_diagnostics,
        fit_summary=fit_summary,
        likelihood_curves=likelihood_curves,
        aggregate_summary=aggregate,
        representative_observed_vs_fitted=representative,
    )
    if config.outdir is not None:
        write_draw_outputs(
            config.outdir,
            counts_by_draw=counts,
            fit_counts_by_draw=fit_counts,
            draw_manifest=draw_manifest,
            site_diagnostics=site_diagnostics,
            draw_diagnostics=draw_diagnostics,
            fit_summary=fit_summary,
            likelihood_curves=likelihood_curves,
            aggregate_summary=aggregate,
            representative_observed_vs_fitted=representative,
            sample_audit=audit,
        )
    return result
