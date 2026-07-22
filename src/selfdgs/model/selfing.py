"""Fast selfing-phase states and probabilities."""

from __future__ import annotations

from collections.abc import Sequence

from selfdgs.model.recursion import State, canonical_state


def fast_state_from_outcome(outcome: Sequence[int | bool], n_diploids: int) -> State:
    """Build the initial slow-coalescent state from fast-collapse outcomes.

    ``outcome[i]`` is true if the two homologs of diploid individual ``i``
    collapse in the fast phase; otherwise they remain separate lineages.
    """
    if len(outcome) != n_diploids:
        raise ValueError(
            f"fast-collapse outcome has length {len(outcome)}, expected {n_diploids}."
        )

    lineages = []
    for individual, collapsed in enumerate(outcome):
        h0 = 2 * individual
        h1 = h0 + 1
        if collapsed:
            lineages.append(frozenset({h0, h1}))
        else:
            lineages.append(frozenset({h0}))
            lineages.append(frozenset({h1}))

    return canonical_state(lineages)


def fast_outcome_probability(outcome: Sequence[int | bool], F: float) -> float:
    """Return the probability of a fast-collapse outcome under inbreeding ``F``."""
    if not (0.0 <= F <= 1.0):
        raise ValueError("F must be between 0 and 1.")

    probability = 1.0
    for collapsed in outcome:
        probability *= F if collapsed else (1.0 - F)
    return probability

