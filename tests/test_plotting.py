from collections import Counter
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from selfdgs.results import FitResult, LikelihoodPoint
from selfdgs.validation import load_likelihood_collection

pytestmark = pytest.mark.plot

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from selfdgs.plotting import (  # noqa: E402
    plot_best_s_distribution,
    plot_cumulative_convergence,
    plot_dgs_counts,
    plot_grouped_likelihood_summary,
    plot_joint_likelihood,
    plot_likelihood_curve,
    plot_likelihood_curves,
    plot_representative_observed_vs_fitted,
    plot_replicate_estimates,
)


def _fit_result() -> FitResult:
    return FitResult(
        best_s=0.5,
        best_loglik=-10.0,
        n_sites=10,
        n_diploids=2,
        method="grid",
        likelihood_curve=(
            LikelihoodPoint(s=0.0, loglik=-12.0, delta_loglik=-2.0),
            LikelihoodPoint(s=0.5, loglik=-10.0, delta_loglik=0.0),
            LikelihoodPoint(s=1.0, loglik=-11.0, delta_loglik=-1.0),
        ),
    )


def test_import_selfdgs_does_not_import_matplotlib():
    code = "import sys; import selfdgs; print('matplotlib' in sys.modules)"
    source = str(Path(__file__).parents[1] / "src")
    pythonpath = os.pathsep.join(filter(None, [source, os.environ.get("PYTHONPATH")]))
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
    )
    assert completed.stdout.strip() == "False"


def test_plot_likelihood_curve_writes_file(tmp_path):
    out = tmp_path / "likelihood.png"

    fig = plot_likelihood_curve(_fit_result(), out=out, true_s=0.5)

    assert out.exists()
    assert out.stat().st_size > 0
    assert fig.axes[0].get_ylabel() == "Delta log-likelihood"


def test_plot_likelihood_curves_and_joint_likelihood_write_files(tmp_path):
    curve_csv = tmp_path / "curve.csv"
    _fit_result().to_csv(curve_csv)

    curves_out = tmp_path / "curves.png"
    joint_out = tmp_path / "joint.png"
    plot_likelihood_curves({"rep1": curve_csv, "rep2": _fit_result()}, out=curves_out)

    joint_frame = pd.concat(
        [
            pd.read_csv(curve_csv).assign(rep=0),
            pd.read_csv(curve_csv).assign(rep=1),
        ],
        ignore_index=True,
    )
    plot_joint_likelihood(joint_frame, out=joint_out, true_s=0.5)

    assert curves_out.exists()
    assert joint_out.exists()
    assert curves_out.stat().st_size > 0
    assert joint_out.stat().st_size > 0


def test_joint_and_cumulative_likelihoods_reject_mismatched_grids():
    mismatched = pd.DataFrame(
        {
            "rep": [0, 0, 1, 1],
            "s": [0.0, 0.5, 0.0, 1.0],
            "loglik": [-2.0, -1.0, -2.0, -1.0],
        }
    )

    with pytest.raises(ValueError, match="identical selfing-rate grids"):
        plot_joint_likelihood(mismatched)
    with pytest.raises(ValueError, match="identical selfing-rate grids"):
        plot_cumulative_convergence(mismatched)


def test_likelihood_summary_plots_reject_duplicate_curve_grid_rows():
    duplicated = pd.DataFrame(
        {
            "group": ["a", "a", "a"],
            "draw_id": [0, 0, 0],
            "rep": [0, 0, 0],
            "s": [0.0, 0.0, 0.5],
            "loglik": [-2.0, -3.0, -1.0],
        }
    )

    with pytest.raises(ValueError, match="exactly one row per curve"):
        plot_joint_likelihood(duplicated)
    with pytest.raises(ValueError, match="exactly one row per curve"):
        plot_cumulative_convergence(duplicated)
    with pytest.raises(ValueError, match="exactly one row per curve"):
        plot_grouped_likelihood_summary(duplicated)


def test_validation_plots_handle_empty_and_missing_estimates():
    estimates = plot_replicate_estimates(
        pd.DataFrame({"best_s": [float("nan"), 0.5]})
    )
    convergence = plot_cumulative_convergence(
        pd.DataFrame(columns=["rep", "s", "loglik"])
    )

    assert sum(patch.get_height() for patch in estimates.axes[0].patches) == 1
    assert len(convergence.axes[0].lines[0].get_xdata()) == 0


def test_plot_dgs_counts_writes_file(tmp_path):
    out = tmp_path / "dgs.png"

    fig = plot_dgs_counts(Counter({(1, 1, 0): 3, (1, 0, 1): 2}), out=out)

    assert out.exists()
    assert out.stat().st_size > 0
    assert fig.axes[0].get_ylabel() == "Count"


def test_validation_plots_and_collection_write_files(tmp_path):
    validation_dir = tmp_path / "validation"
    rep0 = validation_dir / "rep000_seed1"
    rep1 = validation_dir / "rep001_seed2"
    rep0.mkdir(parents=True)
    rep1.mkdir(parents=True)
    _fit_result().to_csv(rep0 / "likelihood.csv")
    _fit_result().to_csv(rep1 / "likelihood.csv")

    likelihoods = load_likelihood_collection([validation_dir])
    assert set(likelihoods["rep"]) == {0, 1}

    summary = pd.DataFrame({"rep": [0, 1, 2], "best_s": [0.4, 0.5, 0.6]})
    estimate_out = tmp_path / "estimates.png"
    convergence_out = tmp_path / "convergence.png"

    plot_replicate_estimates(summary, out=estimate_out, true_s=0.5)
    plot_cumulative_convergence(likelihoods, out=convergence_out, true_s=0.5)

    assert estimate_out.exists()
    assert convergence_out.exists()
    assert estimate_out.stat().st_size > 0
    assert convergence_out.stat().st_size > 0


def test_empirical_plots_write_files(tmp_path):
    likelihoods = pd.DataFrame(
        [
            {"group": "a", "draw_id": 0, "s": 0.0, "loglik": -2.0, "delta_loglik": -1.0},
            {"group": "a", "draw_id": 0, "s": 0.5, "loglik": -1.0, "delta_loglik": 0.0},
            {"group": "a", "draw_id": 1, "s": 0.0, "loglik": -3.0, "delta_loglik": -2.0},
            {"group": "a", "draw_id": 1, "s": 0.5, "loglik": -1.0, "delta_loglik": 0.0},
            {"group": "b", "draw_id": 0, "s": 0.0, "loglik": -1.0, "delta_loglik": 0.0},
            {"group": "b", "draw_id": 0, "s": 0.5, "loglik": -2.0, "delta_loglik": -1.0},
        ]
    )
    fit_summary = pd.DataFrame(
        {"group": ["a", "a", "b"], "draw_id": [0, 1, 0], "best_s": [0.4, 0.5, 0.1]}
    )
    observed_vs_fitted = pd.DataFrame(
        [
            {
                "group": "a",
                "representative_draw_id": 0,
                "n0": 0,
                "n1": 1,
                "n2": 1,
                "observed_count": 5,
                "fitted_expected_count": 4.5,
            },
            {
                "group": "a",
                "representative_draw_id": 0,
                "n0": 1,
                "n1": 0,
                "n2": 1,
                "observed_count": 2,
                "fitted_expected_count": 2.5,
            },
        ]
    )

    likelihood_out = tmp_path / "grouped_likelihood.png"
    distribution_out = tmp_path / "best_s.png"
    observed_out = tmp_path / "observed_vs_fitted.png"

    plot_grouped_likelihood_summary(likelihoods, out=likelihood_out)
    distribution_figure = plot_best_s_distribution(fit_summary, out=distribution_out)
    plot_representative_observed_vs_fitted(observed_vs_fitted, out=observed_out)

    assert likelihood_out.exists()
    assert distribution_out.exists()
    assert observed_out.exists()
    assert likelihood_out.stat().st_size > 0
    assert distribution_out.stat().st_size > 0
    assert observed_out.stat().st_size > 0
    assert [tick.get_text() for tick in distribution_figure.axes[0].get_xticklabels()] == [
        "a",
        "b",
    ]
