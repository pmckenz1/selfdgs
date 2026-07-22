"""Expected DGS branch lengths and probabilities under partial selfing."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import product
from math import comb, isfinite
from numbers import Integral
from typing import Literal, TypeAlias
import warnings

from selfdgs.model.recursion import (
    fast_class_branch_lengths,
    make_branch_length_recursion,
)
from selfdgs.model.selfing import fast_outcome_probability, fast_state_from_outcome
from selfdgs.spectrum import DGSConfig, all_dgs_configs, fold_dgs_config

PRACTICAL_MAX_DIPLOIDS = 6
ModelMethod: TypeAlias = Literal["combinatorial", "recursive"]
MODEL_METHODS = frozenset({"combinatorial", "recursive"})


def _validate_model_inputs(
    s: float,
    N: float,
    n_diploids: int,
    method: str = "combinatorial",
) -> None:
    if not (0.0 <= s <= 1.0):
        raise ValueError("s must be between 0 and 1.")
    if not isfinite(N) or N <= 0:
        raise ValueError("N must be positive and finite.")
    if n_diploids < 1:
        raise ValueError("n_diploids must be at least 1.")
    if method not in MODEL_METHODS:
        choices = ", ".join(sorted(MODEL_METHODS))
        raise ValueError(f"method must be one of: {choices}.")
    if method == "recursive" and n_diploids > PRACTICAL_MAX_DIPLOIDS:
        warnings.warn(
            "The articulated Expected-DGS recursion may become slow for "
            "n_diploids > "
            f"{PRACTICAL_MAX_DIPLOIDS}.",
            RuntimeWarning,
            stacklevel=3,
        )


def _coerce_model_inputs(
    s: float,
    N: float,
    n_diploids: int,
    method: str,
) -> tuple[float, float, int]:
    if isinstance(s, bool):
        raise ValueError("s must be between 0 and 1.")
    if isinstance(N, bool):
        raise ValueError("N must be positive and finite.")
    if isinstance(n_diploids, bool) or not isinstance(n_diploids, Integral):
        raise ValueError("n_diploids must be an integer.")
    checked_s = float(s)
    checked_N = float(N)
    checked_n_diploids = int(n_diploids)
    _validate_model_inputs(
        s=checked_s,
        N=checked_N,
        n_diploids=checked_n_diploids,
        method=method,
    )
    return checked_s, checked_N, checked_n_diploids


def inbreeding_coefficient(s: float) -> float:
    """Return the equilibrium inbreeding coefficient ``F = s / (2 - s)``."""
    if not (0.0 <= s <= 1.0):
        raise ValueError("s must be between 0 and 1.")
    return s / (2.0 - s)


@lru_cache(maxsize=512)
def _expected_dgs_branch_lengths_cached(
    s: float,
    N: float,
    n_diploids: int,
    use_generation_time_scale: bool,
    method: ModelMethod,
) -> tuple[tuple[DGSConfig, float], ...]:
    F = inbreeding_coefficient(s)
    time_scale = (2.0 - s) * N if use_generation_time_scale else 1.0
    total: Counter[DGSConfig] = Counter()
    if method == "combinatorial":
        for collapsed in range(n_diploids + 1):
            probability = (
                comb(n_diploids, collapsed)
                * F**collapsed
                * (1.0 - F) ** (n_diploids - collapsed)
            )
            branch_lengths = fast_class_branch_lengths(
                n_diploids=n_diploids,
                collapsed_diploids=collapsed,
                time_scale=time_scale,
            )
            for config, value in branch_lengths.items():
                total[config] += probability * value
    else:
        branch_length_recursion = make_branch_length_recursion(
            n_diploids=n_diploids,
            time_scale=time_scale,
        )
        for outcome in product([0, 1], repeat=n_diploids):
            probability = fast_outcome_probability(outcome, F)
            state = fast_state_from_outcome(outcome, n_diploids=n_diploids)
            branch_lengths = branch_length_recursion(state)
            for config, value in branch_lengths.items():
                total[config] += probability * value

    for config in all_dgs_configs(n_diploids=n_diploids, polymorphic_only=True):
        total[config] += 0.0

    return tuple(sorted(total.items()))


def expected_dgs_branch_lengths(
    s: float,
    N: float = 1.0,
    n_diploids: int = 4,
    use_generation_time_scale: bool = False,
    method: ModelMethod = "combinatorial",
) -> Counter[DGSConfig]:
    """Compute expected branch lengths by polymorphic DGS cell.

    Parameters
    ----------
    s
        Selfing rate, with ``0 <= s <= 1``.
    N
        Diploid population size. This only affects the result when
        ``use_generation_time_scale=True``.
    n_diploids
        Number of sampled diploid individuals. The recursive reference method
        emits a warning above 6 because its state space grows rapidly.
    use_generation_time_scale
        If true, use ``time_scale = (2 - s) * N``. If false, use normalized
        coalescent units. The global scale cancels when converting to
        probabilities, but generation-time branch lengths are useful for model
        inspection and future extensions.
    method
        ``"combinatorial"`` (the default) groups label-equivalent fast states
        and uses exact Kingman subset weights. ``"recursive"`` runs the
        previous fully articulated state-by-state implementation, retained as
        a readable reference and correctness check.
    """
    s, N, n_diploids = _coerce_model_inputs(s, N, n_diploids, method)
    return Counter(
        dict(
            _expected_dgs_branch_lengths_cached(
                s=s,
                N=N,
                n_diploids=n_diploids,
                use_generation_time_scale=bool(use_generation_time_scale),
                method=method,
            )
        )
    )


@lru_cache(maxsize=512)
def _dgs_probabilities_cached(
    s: float,
    N: float,
    n_diploids: int,
    method: ModelMethod,
) -> tuple[tuple[DGSConfig, float], ...]:
    branch_lengths = expected_dgs_branch_lengths(
        s=s,
        N=N,
        n_diploids=n_diploids,
        use_generation_time_scale=False,
        method=method,
    )
    total_length = sum(branch_lengths.values())
    if total_length <= 0:
        raise ValueError("Total branch length is zero.")
    return tuple(
        (config, branch_lengths[config] / total_length)
        for config in sorted(branch_lengths)
    )


def dgs_probabilities(
    s: float,
    N: float = 1.0,
    n_diploids: int = 4,
    method: ModelMethod = "combinatorial",
) -> dict[DGSConfig, float]:
    """Normalize expected branch lengths into DGS probabilities.

    These probabilities are intended for likelihoods conditional on the number
    of segregating sites. ``N`` is retained in the API for consistency with the
    expected-branch-length function; under the current conditional likelihood,
    it does not change normalized probabilities unless future model options add
    non-global scaling.

    ``method="recursive"`` selects the articulated reference implementation;
    the exact, faster ``"combinatorial"`` implementation is the default.
    """
    s, N, n_diploids = _coerce_model_inputs(s, N, n_diploids, method)
    return dict(
        _dgs_probabilities_cached(
            s=s,
            N=N,
            n_diploids=n_diploids,
            method=method,
        )
    )


@lru_cache(maxsize=512)
def _folded_dgs_probabilities_cached(
    s: float,
    N: float,
    n_diploids: int,
    method: ModelMethod,
) -> tuple[tuple[DGSConfig, float], ...]:
    folded: Counter[DGSConfig] = Counter()
    for config, probability in dgs_probabilities(
        s=s,
        N=N,
        n_diploids=n_diploids,
        method=method,
    ).items():
        folded[fold_dgs_config(config, n_diploids=n_diploids)] += probability
    total_probability = sum(folded.values())
    if total_probability <= 0:
        raise ValueError("Total folded probability is zero.")
    return tuple(
        (config, probability / total_probability)
        for config, probability in sorted(folded.items())
    )


def folded_dgs_probabilities(
    s: float,
    N: float = 1.0,
    n_diploids: int = 4,
    method: ModelMethod = "combinatorial",
) -> dict[DGSConfig, float]:
    """Return model probabilities over folded polymorphic DGS cells.

    This is the appropriate probability vector for unpolarized/folded DGS
    observations, where `(n0, n1, n2)` and `(n2, n1, n0)` represent the same
    allele-orientation class.

    Pass ``method="recursive"`` to use the articulated reference
    implementation instead of the default combinatorial implementation.
    """
    s, N, n_diploids = _coerce_model_inputs(s, N, n_diploids, method)
    return dict(
        _folded_dgs_probabilities_cached(
            s=s,
            N=N,
            n_diploids=n_diploids,
            method=method,
        )
    )
