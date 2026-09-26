"""Validation experiment workflows built around SLiM and existing VCFs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import json
from numbers import Integral, Real
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from selfdgs._outputs import install_staged_outputs
from selfdgs.fit import default_selfing_grid, fit_selfing
from selfdgs.io.vcf import dgs_from_vcf, read_vcf_samples
from selfdgs.results import FitResult
from selfdgs.simulation.slim import (
    SlimRunResult, SlimSimulationConfig, run_slim_simulation, default_slim_script_path,
)
from selfdgs.spectrum import DGSConfig, DGSCounts, filter_polymorphic_dgs, write_dgs_csv


@dataclass(frozen=True)
class VCFResult:
    """Summary from one VCF, which may contain multiple shared-pedigree loci."""

    rep: int
    vcf_index: int
    seed: int | None
    true_s: float | None
    vcf_path: str
    n_all_sites: int
    n_poly_sites: int
    slim_returncode: int | None = None
    slim_command: str | None = None
    slim_stdout_path: str | None = None
    slim_stderr_path: str | None = None
    slim_metadata: str | None = None
    n_loci: int = 1

    @property
    def locus(self) -> int:
        """Alias for vcf_index."""
        return self.vcf_index


# Import alias for callers using the locus-oriented name.
LocusResult = VCFResult


@dataclass(frozen=True)
class ReplicateResult:
    """Summary and fit result from one validation replicate."""

    rep: int
    true_s: float | None
    observed_counts: DGSCounts
    polymorphic_counts: DGSCounts
    fit: FitResult
    vcf_results: tuple[VCFResult, ...]

    @property
    def loci(self) -> tuple[VCFResult, ...]:
        """Alias for vcf_results."""
        return self.vcf_results


@dataclass(frozen=True)
class ValidationExperimentConfig:
    """Configuration for a replicate/locus SLiM validation experiment."""

    slim_script: str | Path | None = None
    outdir: str | Path | None = None
    ne: int | None = field(default=None, repr=False)
    true_s: float = 0.5
    n_independent_loci: int | None = field(default=None, repr=False)
    chrom_length_each: int = 10_000
    mu: float = 1e-7
    recomb_rate: float = 5e-8
    n_sample: int = 4
    burn_mult: int = 10
    n_reps: int = 1
    s_grid: Sequence[float] | None = None
    seed: int = 12345
    slim_executable: str = "slim"
    include_monomorphic: bool = True
    vcf_missing: str = "skip-site"
    vcf_malformed: str = "error"
    vcf_multiallelic: str = "skip"
    vcf_require_pass: bool = True
    vcf_polarization: str = "folded"
    vcf_outgroup_sample_ids: Sequence[str] | None = None
    vcf_min_outgroup_called: int = 1
    vcf_require_homozygous_outgroup: bool = True
    vcf_skip_polymorphic_outgroup: bool = True
    fit_error_rate: float = 0.0
    fit_mode: str | None = None
    extra_defines: Mapping[str, object] = field(default_factory=dict)
    sampling_design: str = "fixed_individuals"
    census_size: int | None = field(default=None, kw_only=True)
    n_loci: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.outdir is None:
            raise ValueError("outdir is required.")
        for name, alias, default in (("census_size", "ne", 500), ("n_loci", "n_independent_loci", 1)):
            value, alias_value = getattr(self, name), getattr(self, alias)
            if value is not None and alias_value is not None and value != alias_value:
                raise ValueError(f"{name} and {alias} must agree when both are supplied.")
            resolved = value if value is not None else alias_value
            resolved = default if resolved is None else resolved
            object.__setattr__(self, name, resolved)
            object.__setattr__(self, alias, resolved)
        if self.sampling_design not in {"fixed_individuals", "independent_populations"}:
            raise ValueError("sampling_design must be fixed_individuals or independent_populations.")
        positive_integers = {
            "census_size": self.census_size,
            "n_loci": self.n_loci,
            "chrom_length_each": self.chrom_length_each,
            "n_sample": self.n_sample,
            "n_reps": self.n_reps,
            "vcf_min_outgroup_called": self.vcf_min_outgroup_called,
        }
        invalid = {
            name: value
            for name, value in positive_integers.items()
            if isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < 1
        }
        if invalid:
            raise ValueError(
                f"Validation integer parameters must be positive integers: {invalid}"
            )
        if self.n_sample > self.census_size:
            raise ValueError("n_sample cannot exceed the diploid census size (census_size).")
        if (
            isinstance(self.burn_mult, bool)
            or not isinstance(self.burn_mult, Integral)
            or self.burn_mult < 0
        ):
            raise ValueError("burn_mult must be a non-negative integer.")
        if self.sampling_design == "fixed_individuals":
            if self.burn_mult < 1:
                raise ValueError("Fixed-individual simulations require burn_mult >= 1.")
            if {"n_loci", "history_path"}.intersection(self.extra_defines):
                raise ValueError("n_loci and history_path are managed by the fixed-individual workflow.")
            if (
                self.slim_script is not None
                and Path(self.slim_script).resolve()
                == default_slim_script_path("independent_populations").resolve()
            ):
                raise ValueError("Use slim_script=None for the bundled fixed-individual model.")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or self.seed < 1
        ):
            raise ValueError("seed must be a positive integer.")
        if (
            isinstance(self.true_s, bool)
            or not isinstance(self.true_s, Real)
            or not np.isfinite(self.true_s)
            or not 0.0 <= self.true_s <= 1.0
        ):
            raise ValueError("true_s must be finite and in [0, 1].")
        for name, value in (("mu", self.mu), ("recomb_rate", self.recomb_rate)):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative.")
        if (
            isinstance(self.fit_error_rate, bool)
            or not isinstance(self.fit_error_rate, Real)
            or not np.isfinite(self.fit_error_rate)
            or not 0.0 <= self.fit_error_rate < 1.0
        ):
            raise ValueError("fit_error_rate must be finite and in [0, 1).")
        _validate_fixed_sample_policies(self.vcf_missing, self.vcf_malformed)
        if self.vcf_multiallelic not in {"skip", "error"}:
            raise ValueError("vcf_multiallelic must be 'skip' or 'error'.")
        if self.vcf_polarization not in {"ref", "folded", "aa", "outgroup-consensus"}:
            raise ValueError("vcf_polarization is invalid.")
        if self.fit_mode not in {None, "unfolded", "folded"}:
            raise ValueError("fit_mode must be 'unfolded', 'folded', or None.")
        grid = [float(value) for value in self.grid()]
        if not grid or any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in grid):
            raise ValueError("s_grid must contain finite values in [0, 1].")

    def grid(self) -> Sequence[float]:
        """Return the selfing-rate grid for fitting."""
        if self.s_grid is None:
            return default_selfing_grid()
        return self.s_grid

    def script_path(self) -> Path:
        """Resolve the model for this sampling design."""
        return Path(self.slim_script) if self.slim_script is not None else default_slim_script_path(self.sampling_design)

    def likelihood_mode(self) -> str:
        """Return the likelihood mode used for fitting parsed DGS counts."""
        if self.fit_mode is not None:
            return self.fit_mode
        return "folded" if self.vcf_polarization == "folded" else "unfolded"


def _fit_counts(
    counts: DGSCounts,
    *,
    ne: int,
    n_sample: int,
    s_grid: Sequence[float],
    fit_error_rate: float = 0.0,
    fit_mode: str = "unfolded",
    input_polarization: str | None = None,
) -> tuple[DGSCounts, FitResult]:
    fit_counts = filter_polymorphic_dgs(counts, n_diploids=n_sample)
    fit = fit_selfing(
        counts,
        N=ne,
        n_diploids=n_sample,
        s_grid=s_grid,
        error_rate=fit_error_rate,
        mode=fit_mode,  # type: ignore[arg-type]
        input_polarization=input_polarization,
    )
    return fit_counts, fit


def _validate_fixed_sample_policies(missing: str, malformed: str) -> None:
    for name, policy in (("missing", missing), ("malformed", malformed)):
        if policy not in {"error", "skip-site"}:
            raise ValueError(
                f"Validation {name} policy must preserve n_sample; got {policy!r}."
            )


def _write_replicate_outputs(replicate: ReplicateResult, rep_outdir: Path) -> None:
    rep_outdir.mkdir(parents=True, exist_ok=True)
    write_dgs_csv(replicate.observed_counts, rep_outdir / "observed_dgs.csv")
    write_dgs_csv(replicate.polymorphic_counts, rep_outdir / "observed_polymorphic_dgs.csv")
    replicate.fit.to_json(rep_outdir / "fit_result.json")
    replicate.fit.to_csv(rep_outdir / "likelihood.csv")
    pd.DataFrame([replicate.fit.summary_dict()]).to_csv(rep_outdir / "fit_summary.csv", index=False)
    pd.DataFrame([asdict(locus) for locus in replicate.vcf_results]).to_csv(
        rep_outdir / "vcfs.csv",
        index=False,
    )


def _write_validation_config(config: ValidationExperimentConfig, outdir: Path) -> None:
    """Persist the complete requested and resolved experiment configuration."""
    payload = asdict(config)
    payload.pop("ne")
    payload.pop("n_independent_loci")
    payload["slim_script"] = str(config.script_path())
    payload["outdir"] = str(config.outdir)
    payload["s_grid"] = None if config.s_grid is None else list(config.s_grid)
    payload["resolved_s_grid"] = [float(value) for value in config.grid()]
    payload["resolved_fit_mode"] = config.likelihood_mode()
    outdir.joinpath("validation_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )


def run_validation_experiment(config: ValidationExperimentConfig) -> tuple[ReplicateResult, ...]:
    """Fit independently simulated replicates under an explicit sampling design.

    Fixed individuals share one multilocus population per replicate. The
    independent_populations control pools a separate population per locus.
    """
    final_outdir = Path(config.outdir)
    if final_outdir.exists() and not final_outdir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {final_outdir}")
    if final_outdir.is_symlink():
        raise ValueError(f"Output directory cannot be a symbolic link: {final_outdir}")
    final_outdir.parent.mkdir(parents=True, exist_ok=True)
    temporary = TemporaryDirectory(
        prefix=f".{final_outdir.name}.",
        dir=final_outdir.parent,
    )
    outdir = Path(temporary.name) / "output"
    outdir.mkdir()
    _write_validation_config(config, outdir)
    rng = np.random.default_rng(config.seed)
    results: list[ReplicateResult] = []

    for rep in range(config.n_reps):
        rep_outdir = outdir / f"rep{rep:03d}"
        total_counts: DGSCounts = Counter()
        loci: list[VCFResult] = []

        fixed = config.sampling_design == "fixed_individuals"
        for locus in range(1 if fixed else config.n_loci):
            locus_seed = int(rng.integers(1, 1_000_000_000))
            vcf_path = rep_outdir / ("sample.vcf" if fixed else f"locus{locus:04d}_seed{locus_seed}.vcf")
            final_vcf_path = final_outdir / rep_outdir.name / vcf_path.name
            extra_defines = dict(config.extra_defines)
            history_path = rep_outdir / "sampled_individuals.csv"
            if fixed:
                extra_defines.update(n_loci=config.n_loci, history_path=history_path)
            slim_config = SlimSimulationConfig(
                slim_script=config.script_path(),
                vcf_path=vcf_path,
                census_size=config.census_size,
                selfing_rate=config.true_s,
                mu=config.mu,
                chrom_length=config.chrom_length_each,
                recomb_rate=config.recomb_rate,
                n_sample=config.n_sample,
                burn_mult=config.burn_mult,
                seed=locus_seed,
                slim_executable=config.slim_executable,
                extra_defines=extra_defines,
            )
            slim_result: SlimRunResult = run_slim_simulation(slim_config)
            final_slim_metadata = dict(slim_result.metadata)
            final_slim_metadata["vcf_path"] = str(final_vcf_path)
            if fixed:
                samples = read_vcf_samples(vcf_path)
                histories = pd.read_csv(history_path)
                required = {"vcf_column", "population_index", "selfing_generations"}
                if not required.issubset(histories.columns):
                    raise ValueError("Invalid fixed-individual sample manifest: missing columns.")
                numeric = histories[list(required)].apply(pd.to_numeric, errors="coerce")
                if (numeric.isna().any().any()
                    or not (numeric % 1 == 0).all().all()
                    or len(samples) != config.n_sample
                    or histories["vcf_column"].tolist() != list(range(config.n_sample))
                    or histories["population_index"].nunique() != config.n_sample
                    or not histories["population_index"].between(0, config.census_size - 1).all()
                    or not histories["selfing_generations"].between(
                        -1, config.burn_mult * config.census_size - 1
                    ).all()):
                    raise ValueError("Invalid fixed-individual sample manifest.")
                histories["vcf_sample"] = samples
                histories["rep"] = rep
                histories["seed"] = locus_seed
                histories.to_csv(history_path, index=False)
                final_slim_metadata["extra_defines"] = {
                    **final_slim_metadata["extra_defines"],
                    "history_path": str(final_outdir / rep_outdir.name / history_path.name),
                }
            stdout_path = vcf_path.with_suffix(".slim.stdout.txt")
            stderr_path = vcf_path.with_suffix(".slim.stderr.txt")
            stdout_path.write_text(slim_result.stdout)
            stderr_path.write_text(slim_result.stderr)
            observed = dgs_from_vcf(
                vcf_path,
                n_diploids=config.n_sample,
                include_monomorphic=config.include_monomorphic,
                missing=config.vcf_missing,
                malformed=config.vcf_malformed,
                multiallelic=config.vcf_multiallelic,
                require_pass=config.vcf_require_pass,
                polarization=config.vcf_polarization,
                outgroup_sample_ids=config.vcf_outgroup_sample_ids,
                min_outgroup_called=config.vcf_min_outgroup_called,
                require_homozygous=config.vcf_require_homozygous_outgroup,
                skip_polymorphic_outgroup=config.vcf_skip_polymorphic_outgroup,
            )
            total_counts.update(observed)
            poly_counts = filter_polymorphic_dgs(observed, n_diploids=config.n_sample)
            loci.append(
                VCFResult(
                    rep=rep,
                    vcf_index=locus,
                    seed=locus_seed,
                    true_s=config.true_s,
                    vcf_path=str(final_vcf_path),
                    n_all_sites=sum(observed.values()),
                    n_poly_sites=sum(poly_counts.values()),
                    n_loci=config.n_loci if fixed else 1,
                    slim_returncode=slim_result.returncode,
                    slim_command=json.dumps(list(slim_result.command)),
                    slim_stdout_path=str(final_vcf_path.with_suffix(".slim.stdout.txt")),
                    slim_stderr_path=str(final_vcf_path.with_suffix(".slim.stderr.txt")),
                    slim_metadata=json.dumps(
                        final_slim_metadata,
                        sort_keys=True,
                        default=str,
                    ),
                )
            )

        polymorphic_counts, fit = _fit_counts(
            total_counts,
            ne=config.census_size,
            n_sample=config.n_sample,
            s_grid=config.grid(),
            fit_error_rate=config.fit_error_rate,
            fit_mode=config.likelihood_mode(),
            input_polarization=config.vcf_polarization,
        )
        fit = replace(fit, metadata={
            **fit.metadata, "sampling_design": config.sampling_design,
            "population_replicate": rep, "n_loci": config.n_loci,
        })
        replicate = ReplicateResult(
            rep=rep,
            true_s=config.true_s,
            observed_counts=Counter(total_counts),
            polymorphic_counts=polymorphic_counts,
            fit=fit,
            vcf_results=tuple(loci),
        )
        _write_replicate_outputs(replicate, rep_outdir)
        results.append(replicate)

    summary_rows = [
        {
            "rep": result.rep,
            "census_size": config.census_size,
            "true_s": result.true_s,
            "n_diploids": config.n_sample,
            "best_s": result.fit.best_s,
            "best_loglik": result.fit.best_loglik,
            "n_sites": result.fit.n_sites,
            "n_loci": config.n_loci,
            "sampling_design": config.sampling_design,
            "chrom_length_each": config.chrom_length_each,
            "mu": config.mu,
            "recomb_rate": config.recomb_rate,
            "burn_mult": config.burn_mult,
            "seed": config.seed,
            "polarization": config.vcf_polarization,
            "missing": config.vcf_missing,
            "malformed": config.vcf_malformed,
            "multiallelic": config.vcf_multiallelic,
            "require_pass": config.vcf_require_pass,
            "fit_error_rate": config.fit_error_rate,
            "fit_mode": config.likelihood_mode(),
            "grid_size": len(config.grid()),
            "boundary_optimum": result.fit.boundary_optimum,
        }
        for result in results
    ]
    pd.DataFrame(summary_rows).to_csv(outdir / "validation_summary.csv", index=False)
    previous_replicates = (
        {
            path.name
            for path in final_outdir.iterdir()
            if path.is_dir() and re.fullmatch(r"rep\d.*", path.name)
        }
        if final_outdir.exists()
        else set()
    )
    owned_names = {
        "validation_summary.csv",
        "validation_config.json",
        *(path.name for path in outdir.iterdir()),
        *previous_replicates,
    }
    install_staged_outputs(outdir, final_outdir, owned_names=owned_names)
    temporary.cleanup()
    return tuple(results)


def summarize_existing_vcfs(
    vcf_paths: Iterable[str | Path],
    *,
    true_s: float | None = None,
    census_size: int | None = None,
    ne: int | None = None,
    n_diploids: int = 4,
    s_grid: Sequence[float] | None = None,
    missing: str = "skip-site",
    malformed: str = "error",
    multiallelic: str = "skip",
    require_pass: bool = True,
    polarization: str = "folded",
    outgroup_sample_ids: Sequence[str] | None = None,
    min_outgroup_called: int = 1,
    require_homozygous_outgroup: bool = True,
    skip_polymorphic_outgroup: bool = True,
    fit_error_rate: float = 0.0,
    fit_mode: str | None = None,
    sampling_design: str = "fixed_individuals",
) -> ReplicateResult:
    """Pool VCFs, checking shared focal identities for fixed individuals.

    Use independent_populations only when different samples across VCFs are
    intentional. Matching names alone cannot establish biological provenance.
    """
    if census_size is not None and ne is not None and census_size != ne:
        raise ValueError("census_size and ne must agree when both are supplied.")
    census_size = census_size if census_size is not None else ne
    census_size = 1 if census_size is None else census_size
    if sampling_design not in {"fixed_individuals", "independent_populations"}:
        raise ValueError("Invalid sampling_design.")
    _validate_fixed_sample_policies(missing, malformed)
    total_counts: DGSCounts = Counter()
    loci: list[VCFResult] = []
    expected_samples = None
    for locus, path in enumerate(vcf_paths):
        if sampling_design == "fixed_individuals":
            samples = set(read_vcf_samples(path))
            if polarization == "outgroup-consensus":
                samples.difference_update(outgroup_sample_ids or ())
            if expected_samples is not None and samples != expected_samples:
                raise ValueError("Fixed-individual VCFs must contain the same focal sample IDs.")
            expected_samples = samples
        observed = dgs_from_vcf(
            path,
            n_diploids=n_diploids,
            missing=missing,
            malformed=malformed,
            multiallelic=multiallelic,
            require_pass=require_pass,
            polarization=polarization,
            outgroup_sample_ids=outgroup_sample_ids,
            min_outgroup_called=min_outgroup_called,
            require_homozygous=require_homozygous_outgroup,
            skip_polymorphic_outgroup=skip_polymorphic_outgroup,
        )
        total_counts.update(observed)
        poly_counts = filter_polymorphic_dgs(observed, n_diploids=n_diploids)
        loci.append(
            VCFResult(
                rep=0,
                vcf_index=locus,
                seed=None,
                true_s=true_s,
                vcf_path=str(path),
                n_all_sites=sum(observed.values()),
                n_poly_sites=sum(poly_counts.values()),
            )
        )

    polymorphic_counts, fit = _fit_counts(
        total_counts,
        ne=census_size,
        n_sample=n_diploids,
        s_grid=default_selfing_grid() if s_grid is None else s_grid,
        fit_error_rate=fit_error_rate,
        fit_mode=fit_mode or ("folded" if polarization == "folded" else "unfolded"),
        input_polarization=polarization,
    )
    fit = replace(fit, metadata={**fit.metadata, "sampling_design": sampling_design})
    return ReplicateResult(
        rep=0,
        true_s=true_s,
        observed_counts=Counter(total_counts),
        polymorphic_counts=polymorphic_counts,
        fit=fit,
        vcf_results=tuple(loci),
    )
