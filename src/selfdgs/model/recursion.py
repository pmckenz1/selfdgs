"""Kingman branch lengths for DGS classes.

The combinatorial implementation is used by default by the public model API.
The fully articulated state-by-state recursion remains here as a readable
reference implementation and as an independent correctness check.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations
from math import comb
from typing import FrozenSet, Iterable, TypeAlias

from selfdgs.spectrum import DGSConfig

Lineage: TypeAlias = FrozenSet[int]
State: TypeAlias = tuple[Lineage, ...]


def genotype_config(descendant_haplotypes: Iterable[int], n_diploids: int) -> DGSConfig:
    """Convert a descendant haplotype set into a DGS cell.

    Haplotypes are indexed as pairs within diploids: individual 0 has
    haplotypes 0 and 1, individual 1 has haplotypes 2 and 3, and so on.
    A mutation on a branch ancestral to ``descendant_haplotypes`` produces
    derived alleles exactly in those haplotypes.
    """
    descendants = set(descendant_haplotypes)
    counts = [0, 0, 0]
    for individual in range(n_diploids):
        h0 = 2 * individual
        h1 = h0 + 1
        derived_count = int(h0 in descendants) + int(h1 in descendants)
        counts[derived_count] += 1
    return (counts[0], counts[1], counts[2])


def canonical_state(lineages: Iterable[Iterable[int]]) -> State:
    """Normalize a coalescent state for memoization."""
    return tuple(
        sorted(
            (frozenset(lineage) for lineage in lineages),
            key=lambda lineage: tuple(sorted(lineage)),
        )
    )


def make_branch_length_recursion(
    n_diploids: int,
    time_scale: float = 1.0,
):
    """Return a memoized expected branch-length recursion.

    The returned function maps a coalescent state to a ``Counter`` from DGS
    cells to expected total branch length subtending those cells. The recursion
    averages over all pairwise coalescences in the slow phase.

    ``time_scale=1`` gives coalescent-unit branch lengths. If
    ``time_scale=(2 - s) * N``, branch lengths are in generations under the
    package's partial-selfing scaling.
    """

    @lru_cache(maxsize=None)
    def branch_lengths(state: State) -> Counter[DGSConfig]:
        state = tuple(state)
        m = len(state)
        out: Counter[DGSConfig] = Counter()

        if m <= 1:
            return out

        tau = time_scale / comb(m, 2)
        for lineage in state:
            config = genotype_config(lineage, n_diploids=n_diploids)
            if config not in {(n_diploids, 0, 0), (0, 0, n_diploids)}:
                out[config] += tau

        pair_count = comb(m, 2)
        for a, b in combinations(range(m), 2):
            merged = state[a] | state[b]
            new_lineages = [state[i] for i in range(m) if i not in (a, b)]
            new_lineages.append(merged)
            child = branch_lengths(canonical_state(new_lineages))
            for config, value in child.items():
                out[config] += value / pair_count

        return out

    return branch_lengths


def fast_class_branch_lengths(
    n_diploids: int,
    collapsed_diploids: int,
    time_scale: float = 1.0,
) -> Counter[DGSConfig]:
    """Return branch lengths for one fast-collapse symmetry class.

    A fast state with ``collapsed_diploids=k`` has ``k`` lineages containing
    both homologs of an individual and ``2 * (n_diploids - k)`` singleton
    lineages. Individual labels do not affect DGS counts, so one calculation
    represents all ``comb(n_diploids, k)`` labeled fast outcomes.

    For ``m`` exchangeable Kingman lineages, every specified subset of ``j``
    initial lineages has expected branch length

    ``2 * time_scale / (j * comb(m, j))``.

    The loops below count, without enumerating them, how many such subsets
    produce each DGS cell. This replaces the full coalescent-state recursion
    with a small polynomial-time sum.
    """
    if n_diploids < 1:
        raise ValueError("n_diploids must be at least 1.")
    if not (0 <= collapsed_diploids <= n_diploids):
        raise ValueError(
            "collapsed_diploids must be between 0 and n_diploids."
        )

    return Counter(
        {
            config: unit_length * time_scale
            for config, unit_length in _fast_class_branch_lengths_unit(
                n_diploids,
                collapsed_diploids,
            )
        }
    )


@lru_cache(maxsize=None)
def _fast_class_branch_lengths_unit(
    n_diploids: int,
    collapsed_diploids: int,
) -> tuple[tuple[DGSConfig, float], ...]:
    """Cache unit-scale class lengths, which are reused across selfing rates."""

    k = collapsed_diploids
    uncollapsed = n_diploids - k
    lineage_count = 2 * n_diploids - k
    out: Counter[DGSConfig] = Counter()

    # c: selected collapsed lineages (each contributes a derived homozygote)
    # h: uncollapsed individuals for which exactly one homolog is selected
    # d: uncollapsed individuals for which both homologs are selected
    for c in range(k + 1):
        collapsed_choices = comb(k, c)
        for h in range(uncollapsed + 1):
            for d in range(uncollapsed - h + 1):
                ancestral = uncollapsed - h - d
                subset_size = c + h + 2 * d
                if subset_size in {0, lineage_count}:
                    continue

                individual_allocations = comb(uncollapsed, h) * comb(
                    uncollapsed - h, d
                )
                subset_count = (
                    collapsed_choices * individual_allocations * (2**h)
                )
                branch_length = 2.0 / (
                    subset_size * comb(lineage_count, subset_size)
                )
                config = (k - c + ancestral, h, c + d)
                out[config] += subset_count * branch_length

    return tuple(sorted(out.items()))
