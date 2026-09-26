"""Selfing-rate estimation from DGS counts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
import warnings

import numpy as np

from selfdgs.likelihood import LikelihoodMode, log_likelihood_dgs
from selfdgs.likelihood import (
    _counts_for_likelihood,
    _log_likelihood_from_probabilities,
    _validate_likelihood_mode,
    _validate_regularization,
    model_probabilities_for_mode,
)
from selfdgs.results import FitResult, LikelihoodPoint
from selfdgs.spectrum import (
    DGSConfig,
    DGSCounts,
    filter_polymorphic_dgs,
    validate_dgs_counts,
)


class AllNonfiniteLikelihoodError(ValueError):
    """Raised when no requested selfing-rate value has a finite likelihood."""

    def __init__(
        self,
        *,
        unsupported_cells: Mapping[DGSConfig, int],
        mode: LikelihoodMode,
    ) -> None:
        self.unsupported_cells = dict(unsupported_cells)
        self.mode = mode
        cells = ", ".join(
            f"{config} (count={count})"
            for config, count in sorted(self.unsupported_cells.items())
        )
        detail = f" Unsupported cells: {cells}." if cells else ""
        super().__init__(
            "all selfing-rate grid likelihoods are non-finite under the strict "
            f"{mode} model.{detail} Exclude cells outside the conditional model "
            "or explicitly set epsilon/error_rate after choosing an appropriate "
            "observation model."
        )


def _regularization_name(*, epsilon: float, error_rate: float) -> str:
    if epsilon > 0.0 and error_rate > 0.0:
        return "uniform_error_mixture+epsilon_floor"
    if error_rate > 0.0:
        return "uniform_error_mixture"
    if epsilon > 0.0:
        return "epsilon_floor"
    return "none"


def _validate_input_polarization(
    *,
    mode: LikelihoodMode,
    input_polarization: str | None,
) -> None:
    if input_polarization == "folded" and mode != "folded":
        raise ValueError(
            "folded input counts must be fitted with the folded likelihood mode"
        )


def _unsupported_metadata(
    unsupported: Mapping[DGSConfig, int],
) -> list[dict[str, int]]:
    return [
        {"n0": config[0], "n1": config[1], "n2": config[2], "count": int(count)}
        for config, count in sorted(unsupported.items())
    ]


def _prepare_fit_counts(
    observed_counts: Mapping[DGSConfig, int],
    *,
    n_diploids: int,
) -> tuple[DGSCounts, DGSCounts, int]:
    """Validate counts and condition them on polymorphic sites for fitting."""
    observed = validate_dgs_counts(observed_counts, n_diploids=n_diploids)
    fit_counts = filter_polymorphic_dgs(observed, n_diploids=n_diploids)
    if not fit_counts:
        raise ValueError(
            "no polymorphic sites remain for the conditional DGS likelihood"
        )
    excluded_cells = sum(
        1 for config, count in observed.items() if count > 0 and config not in fit_counts
    )
    return observed, fit_counts, excluded_cells


def default_selfing_grid(num: int = 1000, upper: float = 0.999) -> np.ndarray:
    """Return the default selfing-rate grid."""
    if num < 2:
        raise ValueError("num must be at least 2.")
    if not (0.0 < upper <= 1.0):
        raise ValueError("upper must be in (0, 1].")
    return np.linspace(0.0, upper, num)


def likelihood_support_interval(
    points: Sequence[LikelihoodPoint],
    drop: float = 1.92,
) -> tuple[float, float] | None:
    """Return the grid support interval within ``drop`` log-likelihood units.

    The default ``drop=1.92`` specifies a descriptive support set, not a
    calibrated confidence interval. Shared individual selfing histories can
    induce dependence even across unlinked loci, in addition to SNP linkage.
    """
    if not math.isfinite(drop) or drop < 0:
        raise ValueError("drop must be finite and non-negative.")
    if not points:
        return None
    supported = [
        point.s
        for point in points
        if math.isfinite(point.delta_loglik) and point.delta_loglik >= -drop
    ]
    if not supported:
        return None
    return (min(supported), max(supported))


def _curve_with_deltas(rows: Sequence[tuple[float, float]]) -> tuple[LikelihoodPoint, ...]:
    finite_logliks = [loglik for _, loglik in rows if math.isfinite(loglik)]
    max_loglik = max(finite_logliks) if finite_logliks else -np.inf
    return tuple(
        LikelihoodPoint(
            s=float(s),
            loglik=float(loglik),
            delta_loglik=float(loglik - max_loglik) if math.isfinite(loglik) else -np.inf,
        )
        for s, loglik in rows
    )


def _boundary_optimum(best_s: float, s_grid: Sequence[float]) -> bool:
    if not s_grid:
        return False
    return best_s == min(s_grid) or best_s == max(s_grid)


def grid_search_selfing(
    observed_counts: Mapping[DGSConfig, int],
    N: float = 1.0,
    n_diploids: int = 4,
    s_grid: Iterable[float] | None = None,
    epsilon: float = 0.0,
    error_rate: float = 0.0,
    support_drop: float = 1.92,
    mode: LikelihoodMode = "unfolded",
    input_polarization: str | None = None,
) -> FitResult:
    """Fit selfing rate by evaluating a fixed selfing-rate grid.

    The DGS likelihood is conditional on segregating sites, so monomorphic
    configurations are always excluded before fitting.
    """
    checked_mode = _validate_likelihood_mode(mode)
    _validate_input_polarization(
        mode=checked_mode,
        input_polarization=input_polarization,
    )
    input_counts, observed, excluded_cells = _prepare_fit_counts(
        observed_counts,
        n_diploids=n_diploids,
    )
    likelihood_observed = _counts_for_likelihood(
        observed,
        n_diploids=n_diploids,
        mode=checked_mode,
    )
    grid = [float(s) for s in (default_selfing_grid() if s_grid is None else s_grid)]
    if not grid:
        raise ValueError("s_grid must contain at least one value.")
    if any(not (0.0 <= s <= 1.0) for s in grid):
        raise ValueError("all values in s_grid must be between 0 and 1.")
    _validate_regularization(epsilon, error_rate)
    if not math.isfinite(support_drop) or support_drop < 0:
        raise ValueError("support_drop must be finite and non-negative.")

    rows: list[tuple[float, float]] = []
    supported_cells: set[DGSConfig] = set()
    for s in grid:
        probabilities = model_probabilities_for_mode(
            s=s,
            N=N,
            n_diploids=n_diploids,
            mode=checked_mode,
        )
        supported_cells.update(
            config
            for config in likelihood_observed
            if probabilities.get(config, 0.0) > 0.0
        )
        rows.append(
            (
                s,
                _log_likelihood_from_probabilities(
                    likelihood_observed,
                    probabilities,
                    epsilon=epsilon,
                    error_rate=error_rate,
                ),
            )
        )
    unsupported = {
        config: int(count)
        for config, count in sorted(likelihood_observed.items())
        if count > 0 and config not in supported_cells
    }
    if not any(math.isfinite(loglik) for _, loglik in rows):
        raise AllNonfiniteLikelihoodError(
            unsupported_cells=unsupported,
            mode=checked_mode,
        )
    best_s, best_loglik = max(rows, key=lambda row: row[1])
    curve = _curve_with_deltas(rows)
    boundary = _boundary_optimum(best_s, grid)
    result_warnings = ()
    if boundary:
        result_warnings = ("best selfing-rate estimate is on the grid boundary",)

    n_input_sites = int(sum(input_counts.values()))
    n_fit_sites = int(sum(observed.values()))
    n_excluded_sites = n_input_sites - n_fit_sites
    metadata = {
        "N": float(N),
        "epsilon": float(epsilon),
        "error_rate": float(error_rate),
        "support_drop": float(support_drop),
        "uncertainty_calibration": "uncalibrated_composite_likelihood",
        "grid_size": len(grid),
        "mode": checked_mode,
        "probability_regularization": _regularization_name(
            epsilon=epsilon,
            error_rate=error_rate,
        ),
        "unsupported_cells": _unsupported_metadata(unsupported),
        "unsupported_cell_count": len(unsupported),
        "unsupported_site_count": int(sum(unsupported.values())),
        "site_conditioning": "polymorphic",
    }
    if input_polarization is not None:
        metadata["input_polarization"] = str(input_polarization)

    return FitResult(
        best_s=float(best_s),
        best_loglik=float(best_loglik),
        n_sites=int(sum(observed.values())),
        n_diploids=int(n_diploids),
        method="grid",
        likelihood_curve=curve,
        support_interval=likelihood_support_interval(curve, drop=support_drop),
        boundary_optimum=boundary,
        warnings=result_warnings,
        metadata=metadata,
        n_input_sites=n_input_sites,
        n_excluded_sites=n_excluded_sites,
        n_excluded_cells=int(excluded_cells),
    )


def fit_selfing(
    observed_counts: Mapping[DGSConfig, int],
    N: float = 1.0,
    n_diploids: int = 4,
    s_grid: Iterable[float] | None = None,
    epsilon: float = 0.0,
    error_rate: float = 0.0,
    support_drop: float = 1.92,
    refine: bool = False,
    refine_bounds: tuple[float, float] = (0.0, 1.0),
    mode: LikelihoodMode = "unfolded",
    input_polarization: str | None = None,
) -> FitResult:
    """Fit selfing rate with grid search and optional continuous refinement.

    Monomorphic configurations are retained in input accounting but are always
    excluded from the conditional-on-segregating-sites likelihood.
    """
    checked_mode = _validate_likelihood_mode(mode)
    grid_result = grid_search_selfing(
        observed_counts=observed_counts,
        N=N,
        n_diploids=n_diploids,
        s_grid=s_grid,
        epsilon=epsilon,
        error_rate=error_rate,
        support_drop=support_drop,
        mode=checked_mode,
        input_polarization=input_polarization,
    )
    if not refine:
        return grid_result

    from scipy.optimize import minimize_scalar

    requested_lower, requested_upper = refine_bounds
    if not (0.0 <= requested_lower < requested_upper <= 1.0):
        raise ValueError("refine_bounds must satisfy 0 <= lower < upper <= 1.")

    grid_values = sorted({point.s for point in grid_result.likelihood_curve})
    if len(grid_values) < 2:
        raise ValueError("continuous refinement requires at least two distinct grid values")
    if not (requested_lower <= grid_result.best_s <= requested_upper):
        raise ValueError("the grid optimum must fall within refine_bounds")
    best_index = grid_values.index(grid_result.best_s)
    lower_neighbor = grid_values[max(0, best_index - 1)]
    upper_neighbor = grid_values[min(len(grid_values) - 1, best_index + 1)]
    lower = max(requested_lower, grid_values[0], lower_neighbor)
    upper = min(requested_upper, grid_values[-1], upper_neighbor)
    if lower >= upper:
        raise ValueError("refinement bounds do not bracket the grid optimum")

    _, observed, _ = _prepare_fit_counts(
        observed_counts,
        n_diploids=n_diploids,
    )

    def objective(s: float) -> float:
        loglik = log_likelihood_dgs(
            observed_counts=observed,
            s=float(s),
            N=N,
            n_diploids=n_diploids,
            epsilon=epsilon,
            error_rate=error_rate,
            mode=checked_mode,
        )
        return -loglik if math.isfinite(loglik) else np.inf

    optimized = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
    if not optimized.success or not math.isfinite(optimized.fun):
        message = "continuous refinement failed; returning grid result"
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return FitResult(
            **{
                **grid_result.summary_dict(),
                "method": "grid",
                "likelihood_curve": grid_result.likelihood_curve,
                "warnings": (*grid_result.warnings, message),
            }
        )

    refined_s = float(optimized.x)
    refined_loglik = float(-optimized.fun)
    rows = [(point.s, point.loglik) for point in grid_result.likelihood_curve]
    if refined_s not in {s for s, _ in rows}:
        rows.append((refined_s, refined_loglik))
    curve = _curve_with_deltas(sorted(rows))
    best_point = max(curve, key=lambda point: point.loglik)
    best_s = best_point.s
    best_loglik = best_point.loglik

    search_lower = max(requested_lower, grid_values[0])
    search_upper = min(requested_upper, grid_values[-1])
    boundary = math.isclose(best_s, search_lower, abs_tol=1e-8) or math.isclose(
        best_s,
        search_upper,
        abs_tol=1e-8,
    )
    result_warnings = tuple(
        warning
        for warning in grid_result.warnings
        if warning != "best selfing-rate estimate is on the grid boundary"
    )
    if boundary:
        result_warnings = (
            *result_warnings,
            "best selfing-rate estimate is on the refinement boundary",
        )

    cutoff = best_loglik - support_drop

    def cutoff_value(s: float) -> float:
        value = -objective(s) - cutoff
        return value if math.isfinite(value) else -np.inf

    def cutoff_boundary(direction: int) -> float:
        """Find the nearest continuous cutoff crossing from the optimum."""
        endpoint = search_lower if direction < 0 else search_upper
        candidates = sorted(
            {
                endpoint,
                best_s,
                *(value for value in grid_values if search_lower <= value <= search_upper),
            },
            reverse=direction < 0,
        )
        inside_s = best_s
        inside_value = cutoff_value(best_s)
        for outside_s in candidates:
            if (direction < 0 and outside_s >= inside_s) or (
                direction > 0 and outside_s <= inside_s
            ):
                continue
            outside_value = cutoff_value(outside_s)
            if outside_value <= 0.0 <= inside_value:
                lo, hi = sorted((outside_s, inside_s))
                # Bisection tolerates non-finite likelihoods outside model support.
                for _ in range(80):
                    midpoint = (lo + hi) / 2.0
                    midpoint_value = cutoff_value(midpoint)
                    if direction < 0:
                        if midpoint_value >= 0.0:
                            hi = midpoint
                        else:
                            lo = midpoint
                    else:
                        if midpoint_value >= 0.0:
                            lo = midpoint
                        else:
                            hi = midpoint
                return (lo + hi) / 2.0
            inside_s = outside_s
            inside_value = outside_value
        return endpoint

    if support_drop == 0.0:
        support_interval = (best_s, best_s)
    else:
        support_interval = (cutoff_boundary(-1), cutoff_boundary(1))
    existing_s = {s for s, _ in rows}
    for support_s in support_interval:
        if support_s not in existing_s:
            rows.append((support_s, -objective(support_s)))
    curve = _curve_with_deltas(sorted(rows))

    return FitResult(
        best_s=best_s,
        best_loglik=best_loglik,
        n_sites=grid_result.n_sites,
        n_diploids=grid_result.n_diploids,
        method="grid+bounded",
        likelihood_curve=curve,
        support_interval=support_interval,
        boundary_optimum=boundary,
        warnings=result_warnings,
        metadata={
            **grid_result.metadata,
            "refine": True,
            "refine_bounds": (float(requested_lower), float(requested_upper)),
            "refine_interval": (float(lower), float(upper)),
            "refine_success": bool(optimized.success),
            "refine_nfev": int(optimized.nfev),
        },
        n_input_sites=grid_result.n_input_sites,
        n_excluded_sites=grid_result.n_excluded_sites,
        n_excluded_cells=grid_result.n_excluded_cells,
    )
