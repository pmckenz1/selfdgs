"""VCF/BCF conversion to diploid genotype spectrum counts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cyvcf2 import VCF

from selfdgs.spectrum import DGSConfig, DGSCounts, is_monomorphic_config

MissingPolicy = Literal["error", "skip-site", "skip-genotype"]
MalformedPolicy = Literal["error", "skip-site", "skip-genotype"]
MultiallelicPolicy = Literal["skip", "error"]
Polarization = Literal["ref", "folded", "aa", "outgroup-consensus"]
AncestralOrientation = Literal["ref", "alt", "folded"]


class VCFParseError(ValueError):
    """Raised when variant records cannot be converted into DGS counts."""


@dataclass(frozen=True)
class VCFScanResult:
    """Counts and diagnostics from one shared variant-file scan."""

    counts_by_set: dict[Hashable, DGSCounts]
    site_diagnostics: dict[str, int]
    set_diagnostics: dict[Hashable, dict[str, int]]
    sample_ids_by_set: dict[Hashable, tuple[str, ...]]


@dataclass(frozen=True)
class AncestralStateDecision:
    """Per-site ancestral-state decision for DGS orientation."""

    orientation: AncestralOrientation | None
    diagnostic: str


def _open_variant_file(path: str | Path) -> VCF:
    try:
        return VCF(str(path), strict_gt=True)
    except Exception as exc:
        raise VCFParseError(f"Could not open variant file {path}: {exc}") from exc


def read_vcf_samples(path: str | Path) -> list[str]:
    """Return sample IDs from a VCF or BCF header."""
    reader = _open_variant_file(path)
    try:
        samples = list(reader.samples)
        if len(samples) != len(set(samples)):
            raise VCFParseError("VCF header contains duplicate sample IDs.")
        return samples
    finally:
        reader.close()


def _genotype_alt_count(genotype: Sequence[int | bool]) -> int | None:
    """Return the ALT count from one cyvcf2 diploid genotype."""
    alleles = genotype[:-1]  # cyvcf2 appends a phased/unphased boolean.
    if len(alleles) != 2:
        raise VCFParseError(f"Genotype is not diploid: {tuple(alleles)!r}.")
    if any(int(allele) < 0 for allele in alleles):
        return None
    if any(int(allele) not in {0, 1} for allele in alleles):
        raise VCFParseError(
            f"Genotype is not biallelic REF/ALT coded: {tuple(alleles)!r}."
        )
    return sum(int(allele) == 1 for allele in alleles)


def _aa_decision(aa_value: object, *, ref: str, alt: str) -> AncestralStateDecision:
    if aa_value is None or aa_value is True or aa_value in ("", "."):
        return AncestralStateDecision(None, "aa_unusable")
    allele = str(aa_value).split("|", 1)[0].split(",", 1)[0].upper()
    if allele == ref.upper():
        return AncestralStateDecision("ref", "aa_ref")
    if allele == alt.upper():
        return AncestralStateDecision("alt", "aa_alt")
    return AncestralStateDecision(None, "aa_unusable")


def _outgroup_consensus_decision(
    genotypes: Sequence[Sequence[int | bool]],
    outgroup_indices: Sequence[int],
    *,
    min_outgroup_called: int,
    require_homozygous: bool,
    skip_polymorphic_outgroup: bool,
) -> AncestralStateDecision:
    called_counts: list[int] = []
    for index in outgroup_indices:
        try:
            alt_count = _genotype_alt_count(genotypes[index])
        except (IndexError, TypeError, VCFParseError):
            return AncestralStateDecision(None, "outgroup_malformed")
        if alt_count is not None:
            called_counts.append(alt_count)

    if len(called_counts) < min_outgroup_called:
        return AncestralStateDecision(None, "outgroup_missing")
    if require_homozygous and any(count == 1 for count in called_counts):
        return AncestralStateDecision(None, "outgroup_heterozygous")

    ref_copies = sum(2 - count for count in called_counts)
    alt_copies = sum(called_counts)
    if skip_polymorphic_outgroup and ref_copies > 0 and alt_copies > 0:
        return AncestralStateDecision(None, "outgroup_polymorphic")
    if ref_copies == alt_copies:
        return AncestralStateDecision(None, "outgroup_polymorphic")
    if ref_copies > alt_copies:
        return AncestralStateDecision("ref", "outgroup_ancestral_ref")
    return AncestralStateDecision("alt", "outgroup_ancestral_alt")


def _resolve_ancestral_state(
    *,
    polarization: Polarization,
    ref: str,
    alt: str,
    aa_value: object,
    genotypes: Sequence[Sequence[int | bool]],
    outgroup_indices: Sequence[int],
    min_outgroup_called: int,
    require_homozygous: bool,
    skip_polymorphic_outgroup: bool,
) -> AncestralStateDecision:
    if polarization == "ref":
        return AncestralStateDecision("ref", "ref_ancestral")
    if polarization == "folded":
        return AncestralStateDecision("folded", "folded")
    if polarization == "aa":
        return _aa_decision(aa_value, ref=ref, alt=alt)
    return _outgroup_consensus_decision(
        genotypes,
        outgroup_indices,
        min_outgroup_called=min_outgroup_called,
        require_homozygous=require_homozygous,
        skip_polymorphic_outgroup=skip_polymorphic_outgroup,
    )


def _orient_alt_count(alt_count: int, orientation: AncestralOrientation) -> int:
    return 2 - alt_count if orientation == "alt" else alt_count


def _config_from_derived_counts(
    derived_counts: Iterable[int],
    *,
    folded: bool,
) -> DGSConfig:
    counts = Counter(derived_counts)
    if set(counts).difference({0, 1, 2}):
        raise VCFParseError("Diploid derived counts must be 0, 1, or 2.")
    config = (counts[0], counts[1], counts[2])
    return min(config, (config[2], config[1], config[0])) if folded else config


def _validate_scan_options(
    *,
    missing: str,
    malformed: str,
    multiallelic: str,
    polarization: str,
    min_outgroup_called: int,
    max_records: int | None,
    outgroup_sample_ids: Sequence[str] | None,
) -> None:
    per_set_policies = {"skip-genotype", "skip-draw"}
    if missing not in {"error", "skip-site", *per_set_policies}:
        raise ValueError(f"Unsupported missing policy: {missing!r}.")
    if malformed not in {"error", "skip-site", *per_set_policies}:
        raise ValueError(f"Unsupported malformed policy: {malformed!r}.")
    if multiallelic not in {"skip", "error"}:
        raise ValueError(f"Unsupported multiallelic policy: {multiallelic!r}.")
    if polarization not in {"ref", "folded", "aa", "outgroup-consensus"}:
        raise ValueError(f"Unsupported polarization mode: {polarization!r}.")
    if min_outgroup_called < 1:
        raise ValueError("min_outgroup_called must be at least 1.")
    if max_records is not None and max_records < 0:
        raise ValueError("max_records must be non-negative or None.")
    if polarization == "outgroup-consensus" and not outgroup_sample_ids:
        raise ValueError(
            "outgroup_sample_ids are required for outgroup-consensus polarization."
        )


def _resolve_sample_sets(
    sample_names: Sequence[str],
    requested_sets: Mapping[Hashable, tuple[str, ...]] | None,
    *,
    outgroup_ids: tuple[str, ...],
    exclude_outgroups: bool,
    n_diploids: int | None,
) -> tuple[
    dict[Hashable, tuple[str, ...]],
    dict[Hashable, tuple[int, ...]],
    tuple[int, ...],
]:
    if len(sample_names) != len(set(sample_names)):
        raise VCFParseError("VCF header contains duplicate sample IDs.")
    sample_to_index = {sample: index for index, sample in enumerate(sample_names)}
    if len(set(outgroup_ids)) != len(outgroup_ids):
        raise ValueError("outgroup_sample_ids contains duplicate sample IDs.")
    missing_outgroups = [sample for sample in outgroup_ids if sample not in sample_to_index]
    if missing_outgroups:
        raise VCFParseError(
            f"Outgroup samples absent from VCF header: {missing_outgroups[:10]}"
        )

    if requested_sets is None:
        excluded = set(outgroup_ids) if exclude_outgroups else set()
        focal = tuple(sample for sample in sample_names if sample not in excluded)
        resolved_sets = {("all_samples", 0): focal}
    else:
        resolved_sets = dict(requested_sets)

    if not resolved_sets or any(not samples for samples in resolved_sets.values()):
        raise ValueError("At least one non-empty sample set is required.")

    set_indices: dict[Hashable, tuple[int, ...]] = {}
    for key, samples in resolved_sets.items():
        if len(set(samples)) != len(samples):
            raise ValueError(f"Sample set {key!r} contains duplicate sample IDs.")
        if exclude_outgroups:
            overlapping = sorted(set(samples).intersection(outgroup_ids))
            if overlapping:
                raise ValueError(
                    f"Sample set {key!r} contains outgroup samples: {overlapping[:10]}. "
                    "Outgroups cannot also be focal samples."
                )
        missing_samples = [sample for sample in samples if sample not in sample_to_index]
        if missing_samples:
            raise VCFParseError(
                f"Samples absent from VCF header for set {key!r}: {missing_samples[:10]}"
            )
        if n_diploids is not None and len(samples) != n_diploids:
            if len(resolved_sets) == 1:
                raise VCFParseError(
                    f"VCF header has {len(samples)} samples, expected {n_diploids}."
                )
            raise VCFParseError(
                f"Sample set {key!r} has {len(samples)} samples, expected {n_diploids}."
            )
        set_indices[key] = tuple(sample_to_index[sample] for sample in samples)

    outgroup_indices = tuple(sample_to_index[sample] for sample in outgroup_ids)
    return resolved_sets, set_indices, outgroup_indices


def scan_vcf_dgs(
    vcf_path: str | Path,
    *,
    sample_sets: Mapping[Hashable, Sequence[str]] | None = None,
    n_diploids: int | None = None,
    include_monomorphic: bool = True,
    missing: str = "skip-site",
    malformed: str = "error",
    multiallelic: MultiallelicPolicy = "skip",
    polarization: Polarization = "folded",
    outgroup_sample_ids: Sequence[str] | None = None,
    min_outgroup_called: int = 1,
    require_homozygous: bool = True,
    skip_polymorphic_outgroup: bool = True,
    require_pass: bool = True,
    max_records: int | None = None,
) -> VCFScanResult:
    """Scan a VCF or BCF once for one or more fixed sample sets.

    cyvcf2/HTSlib handles file decoding. This function applies selfdgs-specific
    filtering, polarization, missing-data policies, and DGS aggregation.
    """
    _validate_scan_options(
        missing=missing,
        malformed=malformed,
        multiallelic=multiallelic,
        polarization=polarization,
        min_outgroup_called=min_outgroup_called,
        max_records=max_records,
        outgroup_sample_ids=outgroup_sample_ids,
    )
    requested_sets = (
        None
        if sample_sets is None
        else {
            key: tuple(str(sample) for sample in samples)
            for key, samples in sample_sets.items()
        }
    )
    outgroup_ids = tuple(str(sample) for sample in outgroup_sample_ids or ())

    reader = _open_variant_file(vcf_path)
    try:
        resolved_sets, set_indices, outgroup_indices = _resolve_sample_sets(
            reader.samples,
            requested_sets,
            outgroup_ids=outgroup_ids,
            exclude_outgroups=polarization == "outgroup-consensus",
            n_diploids=n_diploids,
        )
        focal_indices = sorted({index for indices in set_indices.values() for index in indices})
        counts_by_set: dict[Hashable, DGSCounts] = {
            key: Counter() for key in resolved_sets
        }
        set_diagnostics: dict[Hashable, Counter[str]] = {
            key: Counter() for key in resolved_sets
        }
        site_diagnostics: Counter[str] = Counter()

        for record in reader:
            if max_records is not None and site_diagnostics["records_seen"] >= max_records:
                break
            site_diagnostics["records_seen"] += 1

            ref = record.REF
            alts = list(record.ALT or ())
            if len(alts) != 1:
                if len(alts) > 1 and multiallelic == "error":
                    raise VCFParseError(
                        f"Multiallelic record at {record.CHROM}:{record.POS}: ALT={alts!r}."
                    )
                site_diagnostics["non_biallelic_snp"] += 1
                continue
            alt = alts[0]
            if not record.is_snp or len(ref) != 1 or len(alt) != 1:
                site_diagnostics["non_biallelic_snp"] += 1
                continue
            if require_pass and record.FILTER not in {None, "PASS", "."}:
                site_diagnostics["nonpass_filter"] += 1
                continue

            try:
                genotypes = record.genotypes
            except Exception as exc:
                if malformed == "error":
                    raise VCFParseError(
                        f"Could not parse genotypes at {record.CHROM}:{record.POS}."
                    ) from exc
                site_diagnostics["malformed_record"] += 1
                continue

            aa_value = record.INFO.get("AA") if polarization == "aa" else None
            decision = _resolve_ancestral_state(
                polarization=polarization,
                ref=ref,
                alt=alt,
                aa_value=aa_value,
                genotypes=genotypes,
                outgroup_indices=outgroup_indices,
                min_outgroup_called=min_outgroup_called,
                require_homozygous=require_homozygous,
                skip_polymorphic_outgroup=skip_polymorphic_outgroup,
            )
            site_diagnostics[decision.diagnostic] += 1
            if decision.orientation is None:
                continue

            parsed: dict[int, int | None | str] = {}
            skip_entire_site = False
            for index in focal_indices:
                try:
                    alt_count = _genotype_alt_count(genotypes[index])
                except (IndexError, TypeError, VCFParseError) as exc:
                    if malformed == "error":
                        raise VCFParseError(
                            f"Malformed genotype at {record.CHROM}:{record.POS}."
                        ) from exc
                    if malformed == "skip-site":
                        site_diagnostics["malformed_genotype_site"] += 1
                        skip_entire_site = True
                        break
                    parsed[index] = "malformed"
                    continue
                if alt_count is None and missing == "error":
                    raise VCFParseError(
                        f"Missing genotype at {record.CHROM}:{record.POS}."
                    )
                if alt_count is None and missing == "skip-site":
                    site_diagnostics["missing_genotype_site"] += 1
                    skip_entire_site = True
                    break
                parsed[index] = alt_count
            if skip_entire_site:
                continue

            retained_for_any_set = False
            for key, indices in set_indices.items():
                values = [parsed[index] for index in indices]
                has_malformed = any(value == "malformed" for value in values)
                has_missing = any(value is None for value in values)
                if has_malformed:
                    set_diagnostics[key]["malformed_genotype"] += 1
                    if malformed != "skip-genotype":
                        continue
                if has_missing:
                    set_diagnostics[key]["missing_genotype"] += 1
                    if missing != "skip-genotype":
                        continue
                called = [
                    int(value)
                    for value in values
                    if value is not None and value != "malformed"
                ]
                if not called:
                    set_diagnostics[key]["empty_genotypes"] += 1
                    continue

                derived_counts = [
                    _orient_alt_count(value, decision.orientation) for value in called
                ]
                config = _config_from_derived_counts(
                    derived_counts,
                    folded=decision.orientation == "folded",
                )
                if is_monomorphic_config(config, n_diploids=len(derived_counts)):
                    set_diagnostics[key]["monomorphic_sites"] += 1
                    if not include_monomorphic:
                        continue
                else:
                    set_diagnostics[key]["polymorphic_sites"] += 1
                counts_by_set[key][config] += 1
                set_diagnostics[key]["retained_sites"] += 1
                retained_for_any_set = True

            if retained_for_any_set:
                site_diagnostics["records_retained_for_at_least_one_sample_set"] += 1

        return VCFScanResult(
            counts_by_set={key: Counter(counts) for key, counts in counts_by_set.items()},
            site_diagnostics=dict(site_diagnostics),
            set_diagnostics={key: dict(values) for key, values in set_diagnostics.items()},
            sample_ids_by_set=dict(resolved_sets),
        )
    except VCFParseError:
        raise
    except Exception as exc:
        raise VCFParseError(f"Could not parse variant file {vcf_path}: {exc}") from exc
    finally:
        reader.close()


def dgs_from_vcf(
    vcf_path: str | Path,
    *,
    n_diploids: int | None = None,
    include_monomorphic: bool = True,
    missing: MissingPolicy = "skip-site",
    malformed: MalformedPolicy = "error",
    multiallelic: MultiallelicPolicy = "skip",
    polarization: Polarization = "folded",
    outgroup_sample_ids: Sequence[str] | None = None,
    min_outgroup_called: int = 1,
    require_homozygous: bool = True,
    skip_polymorphic_outgroup: bool = True,
    require_pass: bool = True,
    max_records: int | None = None,
) -> DGSCounts:
    """Return DGS counts from a VCF or BCF file."""
    result = scan_vcf_dgs(
        vcf_path,
        n_diploids=n_diploids,
        include_monomorphic=include_monomorphic,
        missing=missing,
        malformed=malformed,
        multiallelic=multiallelic,
        polarization=polarization,
        outgroup_sample_ids=outgroup_sample_ids,
        min_outgroup_called=min_outgroup_called,
        require_homozygous=require_homozygous,
        skip_polymorphic_outgroup=skip_polymorphic_outgroup,
        require_pass=require_pass,
        max_records=max_records,
    )
    return Counter(next(iter(result.counts_by_set.values())))
