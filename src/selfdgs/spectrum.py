"""Core DGS data structures and utilities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import TypeAlias

import pandas as pd

DGSConfig: TypeAlias = tuple[int, int, int]
DGSCounts: TypeAlias = Counter[DGSConfig]


class DGSError(ValueError):
    """Raised when DGS data are invalid."""


def validate_dgs_config(config: Sequence[int], n_diploids: int | None = None) -> DGSConfig:
    """Validate and normalize a DGS cell `(n0, n1, n2)`.

    Parameters
    ----------
    config
        Three non-negative integer genotype counts.
    n_diploids
        Optional expected sample size. If provided, the cell must sum to this
        value.
    """
    if len(config) != 3:
        raise DGSError("DGS config must have exactly three entries: (n0, n1, n2).")

    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in config):
        raise DGSError(f"DGS config contains a non-integer value: {config!r}.")
    out = tuple(int(value) for value in config)

    if any(x < 0 for x in out):
        raise DGSError(f"DGS config contains a negative count: {config!r}.")

    if n_diploids is not None and sum(out) != n_diploids:
        raise DGSError(
            f"DGS config {out!r} sums to {sum(out)}, expected {n_diploids}."
        )

    return out  # type: ignore[return-value]


def validate_dgs_counts(
    counts: Mapping[DGSConfig, int],
    n_diploids: int | None = None,
) -> DGSCounts:
    """Validate DGS counts and return a normalized ``Counter``."""
    out: DGSCounts = Counter()
    for config, count in counts.items():
        cell = validate_dgs_config(config, n_diploids=n_diploids)
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise DGSError(f"DGS count for {cell!r} is not an integer: {count!r}.")
        value = int(count)
        if value < 0:
            raise DGSError(f"DGS count for {cell!r} is negative: {value}.")
        if value:
            out[cell] += value
    return out


def is_monomorphic_config(config: Sequence[int], n_diploids: int | None = None) -> bool:
    """Return true if a DGS cell is all ancestral or all derived homozygotes."""
    n0, n1, n2 = validate_dgs_config(config, n_diploids=n_diploids)
    n = n_diploids if n_diploids is not None else n0 + n1 + n2
    return (n0, n1, n2) in {(n, 0, 0), (0, 0, n)}


def fold_dgs_config(config: Sequence[int], n_diploids: int | None = None) -> DGSConfig:
    """Fold a DGS cell by canonicalizing the two allele orientations.

    Folded analyses are used when the ancestral/reference orientation is not
    meaningful. The heterozygote count is unchanged, while the two homozygote
    counts are compared in both orientations and the lexicographically smaller
    cell is returned as the canonical representation.
    """
    n0, n1, n2 = validate_dgs_config(config, n_diploids=n_diploids)
    return min((n0, n1, n2), (n2, n1, n0))


def all_dgs_configs(n_diploids: int, polymorphic_only: bool = True) -> list[DGSConfig]:
    """Enumerate all DGS cells for a sample of ``n_diploids`` individuals."""
    if n_diploids < 1:
        raise DGSError("n_diploids must be at least 1.")

    configs: list[DGSConfig] = []
    for n0 in range(n_diploids + 1):
        for n1 in range(n_diploids - n0 + 1):
            n2 = n_diploids - n0 - n1
            config = (n0, n1, n2)
            if polymorphic_only and is_monomorphic_config(config, n_diploids):
                continue
            configs.append(config)
    return configs


def filter_polymorphic_dgs(
    observed: Mapping[DGSConfig, int],
    n_diploids: int,
) -> DGSCounts:
    """Remove monomorphic DGS cells and zero-count cells."""
    counts = validate_dgs_counts(observed, n_diploids=n_diploids)
    return Counter(
        {
            config: count
            for config, count in counts.items()
            if not is_monomorphic_config(config, n_diploids)
        }
    )


def dgs_to_dataframe(counts: Mapping[DGSConfig, int]) -> pd.DataFrame:
    """Convert one DGS counter to a tidy DataFrame with ``n0``, ``n1``, ``n2``."""
    rows = [
        {"n0": n0, "n1": n1, "n2": n2, "count": count}
        for (n0, n1, n2), count in sorted(validate_dgs_counts(counts).items())
    ]
    return pd.DataFrame(rows, columns=["n0", "n1", "n2", "count"])


def dataframe_to_dgs(frame: pd.DataFrame) -> DGSCounts:
    """Convert a tidy DGS DataFrame with ``n0``, ``n1``, ``n2``, ``count`` columns."""
    required = {"n0", "n1", "n2", "count"}
    missing = required.difference(frame.columns)
    if missing:
        raise DGSError(f"DGS DataFrame is missing required columns: {sorted(missing)}.")

    counts: DGSCounts = Counter()
    for row in frame.loc[:, ["n0", "n1", "n2", "count"]].itertuples(index=False):
        config = validate_dgs_config((row.n0, row.n1, row.n2))
        if isinstance(row.count, bool) or not isinstance(row.count, Integral):
            raise DGSError(f"DGS count for {config!r} is not an integer: {row.count!r}.")
        count = int(row.count)
        if count < 0:
            raise DGSError(f"DGS count for {config!r} is negative: {count}.")
        if count:
            counts[config] += count
    return counts


def counters_to_dataframe(
    counters: Sequence[Mapping[DGSConfig, int]],
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Convert replicate DGS counters to a wide DataFrame indexed by DGS cell."""
    if names is None:
        names = [f"rep{i + 1}" for i in range(len(counters))]
    if len(names) != len(counters):
        raise DGSError("names must have the same length as counters.")

    series = [
        pd.Series(validate_dgs_counts(counter), name=name, dtype="int64")
        for counter, name in zip(counters, names)
    ]
    if not series:
        return pd.DataFrame()

    frame = pd.concat(series, axis=1).fillna(0).astype(int)
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=["n0", "n1", "n2"])
    return frame.sort_index()


def read_dgs_csv(path: str | Path) -> DGSCounts:
    """Read tidy DGS counts from CSV."""
    return dataframe_to_dgs(pd.read_csv(path))


def write_dgs_csv(counts: Mapping[DGSConfig, int], path: str | Path) -> None:
    """Write tidy DGS counts to CSV."""
    dgs_to_dataframe(counts).to_csv(path, index=False)
