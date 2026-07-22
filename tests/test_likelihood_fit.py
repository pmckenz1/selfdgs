from collections import Counter
import json

import numpy as np
import pandas as pd
import pytest

from selfdgs import (
    folded_dgs_probabilities,
    fit_result_from_json,
    fit_selfing,
    grid_search_selfing,
    log_likelihood_dgs,
    unsupported_dgs_cells,
)
from selfdgs.fit import default_selfing_grid, likelihood_support_interval
from selfdgs.model import dgs_probabilities
from selfdgs.results import FitResult, LikelihoodPoint


def test_log_likelihood_matches_manual_calculation():
    observed = Counter({(1, 1, 0): 3, (1, 0, 1): 2})
    probabilities = dgs_probabilities(s=0.5, n_diploids=2)

    expected = (
        3 * np.log(probabilities[(1, 1, 0)])
        + 2 * np.log(probabilities[(1, 0, 1)])
    )

    assert log_likelihood_dgs(observed, s=0.5, n_diploids=2) == pytest.approx(expected)


def test_folded_log_likelihood_matches_folded_manual_calculation():
    observed = Counter({(0, 1, 1): 3, (1, 0, 1): 2})
    probabilities = folded_dgs_probabilities(s=0.5, n_diploids=2)

    expected = (
        3 * np.log(probabilities[(0, 1, 1)])
        + 2 * np.log(probabilities[(1, 0, 1)])
    )

    assert log_likelihood_dgs(
        observed,
        s=0.5,
        n_diploids=2,
        mode="folded",
    ) == pytest.approx(expected)
    assert log_likelihood_dgs(observed, s=0.5, n_diploids=2) != pytest.approx(expected)


def test_folded_likelihood_canonicalizes_unfolded_observed_counts():
    folded = Counter({(0, 1, 1): 4})
    unfolded_oriented = Counter({(0, 1, 1): 1, (1, 1, 0): 3})

    assert log_likelihood_dgs(
        unfolded_oriented,
        s=0.5,
        n_diploids=2,
        mode="folded",
    ) == pytest.approx(
        log_likelihood_dgs(folded, s=0.5, n_diploids=2, mode="folded")
    )


def test_impossible_observed_class_returns_negative_infinity_unless_smoothed():
    observed = Counter({(2, 0, 0): 1})

    assert log_likelihood_dgs(observed, s=0.5, n_diploids=2) == -np.inf
    assert np.isfinite(log_likelihood_dgs(observed, s=0.5, n_diploids=2, epsilon=1e-12))


def test_log_likelihood_rejects_negative_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        log_likelihood_dgs(Counter({(1, 1, 0): 1}), s=0.5, n_diploids=2, epsilon=-1)


@pytest.mark.parametrize("epsilon", [np.nan, np.inf, 1.01])
def test_log_likelihood_rejects_non_probability_epsilon(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        log_likelihood_dgs(
            Counter({(1, 1, 0): 1}),
            s=0.5,
            n_diploids=2,
            epsilon=epsilon,
        )


def test_error_rate_makes_full_selfing_boundary_finite_for_rare_heterozygotes():
    observed = Counter({(2, 0, 2): 100, (3, 0, 1): 200, (3, 1, 0): 1})

    assert log_likelihood_dgs(observed, s=1.0, n_diploids=4) == -np.inf
    assert np.isfinite(
        log_likelihood_dgs(observed, s=1.0, n_diploids=4, error_rate=1e-3)
    )

    result = grid_search_selfing(
        observed,
        n_diploids=4,
        s_grid=[0.0, 0.9, 1.0],
        error_rate=1e-3,
    )

    assert result.best_s == pytest.approx(1.0)
    assert result.metadata["error_rate"] == pytest.approx(1e-3)


def test_error_rate_normalizes_over_model_and_unsupported_observed_cells():
    observed = Counter({(2, 0, 0): 1})
    model = dgs_probabilities(s=0.5, n_diploids=2)
    error_rate = 0.1

    expected = np.log(error_rate / (len(model) + 1))

    assert log_likelihood_dgs(
        observed,
        s=0.5,
        n_diploids=2,
        error_rate=error_rate,
    ) == pytest.approx(expected)


def test_log_likelihood_rejects_invalid_error_rate():
    observed = Counter({(1, 1, 0): 1})

    with pytest.raises(ValueError, match="error_rate"):
        log_likelihood_dgs(observed, s=0.5, n_diploids=2, error_rate=-0.1)
    with pytest.raises(ValueError, match="error_rate"):
        log_likelihood_dgs(observed, s=0.5, n_diploids=2, error_rate=1.0)


def test_log_likelihood_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        log_likelihood_dgs(Counter({(1, 1, 0): 1}), s=0.5, n_diploids=2, mode="bad")  # type: ignore[arg-type]


def test_grid_search_selfing_recovers_synthetic_grid_maximum():
    probabilities = dgs_probabilities(s=0.5, n_diploids=2)
    observed = Counter({config: round(probability * 10_000) for config, probability in probabilities.items()})

    result = grid_search_selfing(
        observed,
        n_diploids=2,
        s_grid=np.linspace(0.0, 1.0, 11),
    )

    assert isinstance(result, FitResult)
    assert result.best_s == pytest.approx(0.5)
    assert result.method == "grid"
    assert result.n_sites == sum(observed.values())
    assert result.n_diploids == 2
    assert result.boundary_optimum is False
    assert result.support_interval == (0.5, 0.5)
    assert min(point.delta_loglik for point in result.likelihood_curve) <= 0
    assert max(point.delta_loglik for point in result.likelihood_curve) == pytest.approx(0.0)


def test_grid_search_selfing_recovers_synthetic_folded_grid_maximum():
    probabilities = folded_dgs_probabilities(s=0.5, n_diploids=2)
    observed = Counter({config: round(probability * 10_000) for config, probability in probabilities.items()})

    result = grid_search_selfing(
        observed,
        n_diploids=2,
        s_grid=np.linspace(0.0, 1.0, 11),
        mode="folded",
        input_polarization="folded",
    )

    assert result.best_s == pytest.approx(0.5)
    assert result.metadata["mode"] == "folded"
    assert result.metadata["input_polarization"] == "folded"


def test_fit_selfing_without_refinement_returns_grid_result():
    observed = Counter({(2, 0, 0): 7, (1, 1, 0): 10, (1, 0, 1): 3})

    result = fit_selfing(observed, n_diploids=2, s_grid=[0.0, 0.5, 1.0], refine=False)

    assert result.method == "grid"
    assert len(result.likelihood_curve) == 3
    assert result.n_sites == 13
    assert result.n_input_sites == 20
    assert result.n_excluded_sites == 7
    assert "n_input_sites" not in result.metadata
    assert "n_fit_sites" not in result.metadata
    assert "n_excluded_sites" not in result.metadata
    assert "n_excluded_cells" not in result.metadata


def test_fit_selfing_refines_locally_and_updates_curve_diagnostics():
    probabilities = dgs_probabilities(s=0.53, n_diploids=2)
    observed = Counter(
        {
            config: round(probability * 100_000)
            for config, probability in probabilities.items()
        }
    )

    result = fit_selfing(
        observed,
        n_diploids=2,
        s_grid=[0.0, 0.5, 1.0],
        refine=True,
    )

    assert result.method == "grid+bounded"
    assert result.best_s == pytest.approx(0.53, abs=1e-3)
    assert result.boundary_optimum is False
    assert result.warnings == ()
    assert any(point.s == result.best_s for point in result.likelihood_curve)
    assert max(point.delta_loglik for point in result.likelihood_curve) == pytest.approx(0.0)
    assert result.metadata["refine_interval"] == (0.0, 1.0)
    assert result.support_interval[0] < result.best_s < result.support_interval[1]
    assert result.support_interval != (result.best_s, result.best_s)
    support_points = {
        point.s: point.delta_loglik for point in result.likelihood_curve
    }
    assert support_points[result.support_interval[0]] == pytest.approx(-1.92)
    assert support_points[result.support_interval[1]] == pytest.approx(-1.92)


def test_unfolded_likelihood_rejects_folded_input_counts():
    observed = Counter({(0, 1, 1): 10, (1, 0, 1): 5})

    with pytest.raises(ValueError, match="folded input counts"):
        grid_search_selfing(
            observed,
            n_diploids=2,
            s_grid=[0.0, 0.5, 1.0],
            mode="unfolded",
            input_polarization="folded",
        )


def test_refinement_handles_a_grid_optimum_at_full_selfing():
    probabilities = dgs_probabilities(s=1.0, n_diploids=2)
    observed = Counter(
        {
            config: round(probability * 100_000)
            for config, probability in probabilities.items()
        }
    )

    result = fit_selfing(
        observed,
        n_diploids=2,
        s_grid=[0.0, 0.5, 1.0],
        refine=True,
    )

    assert result.best_s == 1.0
    assert result.boundary_optimum is True
    assert result.warnings == (
        "best selfing-rate estimate is on the refinement boundary",
    )


def test_grid_search_reports_boundary_optimum():
    observed = Counter({(1, 1, 0): 100})
    result = grid_search_selfing(observed, n_diploids=2, s_grid=[0.0, 0.5, 1.0])

    assert result.boundary_optimum is True
    assert result.warnings == ("best selfing-rate estimate is on the grid boundary",)


def test_grid_search_rejects_input_without_polymorphic_sites():
    observed = Counter({(2, 0, 0): 3})

    with pytest.raises(ValueError, match="no polymorphic sites remain"):
        grid_search_selfing(observed, n_diploids=2, s_grid=[0.0, 0.5, 1.0])


def test_grid_search_excludes_monomorphic_sites_before_regularization():
    observed = Counter({(2, 0, 0): 3, (1, 1, 0): 2})

    result = grid_search_selfing(
        observed,
        n_diploids=2,
        s_grid=[0.0, 0.5, 1.0],
        error_rate=1e-3,
    )

    assert unsupported_dgs_cells(
        observed,
        [0.0, 0.5, 1.0],
        n_diploids=2,
    ) == {(2, 0, 0): 3}
    assert result.metadata["probability_regularization"] == "uniform_error_mixture"
    assert result.n_sites == 2
    assert result.n_input_sites == 5
    assert result.n_excluded_sites == 3
    assert result.n_excluded_cells == 1
    assert result.metadata["unsupported_cells"] == []
    assert result.metadata["unsupported_site_count"] == 0
    assert result.metadata["site_conditioning"] == "polymorphic"


def test_likelihood_support_interval_uses_delta_cutoff():
    points = (
        LikelihoodPoint(s=0.0, loglik=-5.0, delta_loglik=-3.0),
        LikelihoodPoint(s=0.5, loglik=-2.0, delta_loglik=0.0),
        LikelihoodPoint(s=1.0, loglik=-3.0, delta_loglik=-1.0),
    )

    assert likelihood_support_interval(points, drop=1.92) == (0.5, 1.0)

    with pytest.raises(ValueError, match="finite"):
        likelihood_support_interval(points, drop=np.nan)


def test_grid_search_rejects_nonfinite_support_drop():
    with pytest.raises(ValueError, match="support_drop"):
        grid_search_selfing(
            Counter({(1, 1, 0): 1}),
            n_diploids=2,
            s_grid=[0.0, 0.5],
            support_drop=np.nan,
        )


def test_default_selfing_grid_validation():
    assert len(default_selfing_grid(num=5, upper=1.0)) == 5

    with pytest.raises(ValueError, match="num"):
        default_selfing_grid(num=1)

    with pytest.raises(ValueError, match="upper"):
        default_selfing_grid(upper=0.0)


def test_fit_result_serializes_to_json_and_csv(tmp_path):
    result = FitResult(
        best_s=0.5,
        best_loglik=-10.0,
        n_sites=5,
        n_diploids=2,
        method="grid",
        likelihood_curve=(
            LikelihoodPoint(s=0.0, loglik=-np.inf, delta_loglik=-np.inf),
            LikelihoodPoint(s=0.5, loglik=-10.0, delta_loglik=0.0),
        ),
        support_interval=(0.5, 0.5),
        metadata={"N": 1.0},
    )

    json_path = tmp_path / "fit.json"
    csv_path = tmp_path / "likelihood.csv"
    result.to_json(json_path)
    result.to_csv(csv_path)

    payload = json.loads(json_path.read_text())
    assert payload["best_s"] == 0.5
    assert payload["likelihood_curve"][0]["loglik"] is None
    assert payload["likelihood_curve"][0]["delta_loglik"] is None
    assert len(payload["likelihood_curve"]) == 2

    restored = fit_result_from_json(json_path)
    assert restored.best_s == result.best_s
    assert restored.likelihood_curve[0].loglik == -np.inf
    assert restored.support_interval == result.support_interval
    assert restored.likelihood_curve[1].delta_loglik == 0.0

    frame = pd.read_csv(csv_path)
    assert list(frame.columns) == ["s", "loglik", "delta_loglik"]
    assert len(frame) == 2
