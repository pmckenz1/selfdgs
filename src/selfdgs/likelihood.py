"""Composite likelihoods for DGS selfing-rate inference."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from collections import Counter
from math import log
from typing import Literal

import numpy as np

from selfdgs.model import dgs_probabilities, folded_dgs_probabilities
from selfdgs.spectrum import DGSConfig, validate_dgs_counts
from selfdgs.spectrum import fold_dgs_config

LikelihoodMode = Literal["unfolded", "folded"]


def _validate_likelihood_mode(mode: str) -> LikelihoodMode:
    if mode not in {"unfolded", "folded"}:
        raise ValueError("mode must be 'unfolded' or 'folded'.")
    return mode  # type: ignore[return-value]


def _counts_for_likelihood(
    observed_counts: Mapping[DGSConfig, int],
    *,
    n_diploids: int,
    mode: LikelihoodMode,
) -> Counter[DGSConfig]:
    observed = validate_dgs_counts(observed_counts, n_diploids=n_diploids)
    if mode == "unfolded":
        return observed
    folded: Counter[DGSConfig] = Counter()
    for config, count in observed.items():
        folded[fold_dgs_config(config, n_diploids=n_diploids)] += count
    return folded


def model_probabilities_for_mode(
    s: float,
    N: float = 1.0,
    n_diploids: int = 4,
    mode: LikelihoodMode = "unfolded",
) -> dict[DGSConfig, float]:
    """Return expected DGS probabilities for a likelihood mode."""
    checked_mode = _validate_likelihood_mode(mode)
    if checked_mode == "folded":
        return folded_dgs_probabilities(s=s, N=N, n_diploids=n_diploids)
    return dgs_probabilities(s=s, N=N, n_diploids=n_diploids)


def _validate_regularization(epsilon: float, error_rate: float) -> None:
    if not np.isfinite(epsilon) or not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be finite and in [0, 1].")
    if not np.isfinite(error_rate) or not (0.0 <= error_rate < 1.0):
        raise ValueError("error_rate must be in [0, 1).")


def _log_likelihood_from_probabilities(
    observed: Mapping[DGSConfig, int],
    probabilities: Mapping[DGSConfig, float],
    *,
    epsilon: float,
    error_rate: float,
) -> float:
    error_support = set(probabilities).union(
        config for config, count in observed.items() if count > 0
    )
    background = 1.0 / len(error_support) if error_rate > 0.0 else 0.0
    loglik = 0.0
    for config, count in observed.items():
        if count == 0:
            continue
        probability = probabilities.get(config, 0.0)
        if error_rate > 0.0:
            probability = (1.0 - error_rate) * probability + error_rate * background
        if epsilon > 0:
            probability = max(probability, epsilon)
        if probability <= 0.0:
            return -np.inf
        loglik += count * log(probability)
    return loglik


def unsupported_dgs_cells(
    observed_counts: Mapping[DGSConfig, int],
    s_grid: Iterable[float],
    *,
    N: float = 1.0,
    n_diploids: int = 4,
    mode: LikelihoodMode = "unfolded",
) -> dict[DGSConfig, int]:
    """Return positive-count cells with zero probability across a model grid.

    This check describes support under the strict DGS model. It intentionally
    ignores epsilon flooring and observation-error mixtures so regularized fits
    can still report which observations required regularization.
    """
    checked_mode = _validate_likelihood_mode(mode)
    observed = _counts_for_likelihood(
        observed_counts,
        n_diploids=n_diploids,
        mode=checked_mode,
    )
    grid = [float(s) for s in s_grid]
    if not grid:
        raise ValueError("s_grid must contain at least one value.")
    supported: set[DGSConfig] = set()
    for s in grid:
        probabilities = model_probabilities_for_mode(
            s=s,
            N=N,
            n_diploids=n_diploids,
            mode=checked_mode,
        )
        supported.update(config for config in observed if probabilities.get(config, 0.0) > 0.0)
    return {
        config: int(count)
        for config, count in sorted(observed.items())
        if count > 0 and config not in supported
    }


def log_likelihood_dgs(
    observed_counts: Mapping[DGSConfig, int],
    s: float,
    N: float = 1.0,
    n_diploids: int = 4,
    epsilon: float = 0.0,
    error_rate: float = 0.0,
    mode: LikelihoodMode = "unfolded",
) -> float:
    """Evaluate the multinomial composite log-likelihood for DGS counts.

    The multinomial coefficient is omitted because it is constant in ``s``.
    Positive observed counts in DGS cells with zero model probability return
    ``-inf`` unless ``epsilon`` is positive, in which case probabilities are
    floored at ``epsilon``. If ``error_rate`` is positive, the model
    probabilities are mixed with a uniform distribution over the union of
    model cells and positive-count observed cells. This is useful for cases
    where a strict coalescent model assigns zero probability to rare observed
    cells.
    """
    _validate_regularization(epsilon, error_rate)
    checked_mode = _validate_likelihood_mode(mode)

    observed = _counts_for_likelihood(
        observed_counts,
        n_diploids=n_diploids,
        mode=checked_mode,
    )
    probabilities = model_probabilities_for_mode(
        s=s,
        N=N,
        n_diploids=n_diploids,
        mode=checked_mode,
    )
    return _log_likelihood_from_probabilities(
        observed,
        probabilities,
        epsilon=epsilon,
        error_rate=error_rate,
    )
