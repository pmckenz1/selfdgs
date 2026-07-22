"""Empirical grouped-draw plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from selfdgs.plotting._common import dataframe_or_csv, require_pyplot, save_or_return
from selfdgs.plotting.likelihood import _require_identical_s_grids


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} data are missing columns: {sorted(missing)}.")


def _cell_labels(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(lambda row: f"({int(row.n0)},{int(row.n1)},{int(row.n2)})", axis=1)


def plot_grouped_likelihood_summary(
    likelihood_curves: pd.DataFrame | str | Path,
    *,
    out: str | Path | None = None,
    delta: bool = True,
):
    """Plot median draw likelihood curves with interquartile bands by group."""
    plt = require_pyplot()
    frame = dataframe_or_csv(likelihood_curves)
    _require_columns(frame, {"group", "draw_id", "s", "loglik"}, "grouped likelihood")
    _require_identical_s_grids(frame, curve_columns=("group", "draw_id"))
    frame = frame.copy()
    if "delta_loglik" not in frame.columns:
        frame["delta_loglik"] = frame["loglik"] - frame.groupby(["group", "draw_id"])[
            "loglik"
        ].transform("max")
    y_column = "delta_loglik" if delta else "loglik"

    summary = (
        frame.groupby(["group", "s"], as_index=False)[y_column]
        .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .sort_values(["group", "s"])
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    for group, group_frame in summary.groupby("group", sort=True):
        x = group_frame["s"].to_numpy(dtype=float)
        median = group_frame["median"].to_numpy(dtype=float)
        q25 = group_frame["q25"].to_numpy(dtype=float)
        q75 = group_frame["q75"].to_numpy(dtype=float)
        ax.plot(x, median, linewidth=1.5, label=str(group))
        ax.fill_between(x, q25, q75, alpha=0.18)
    ax.set_xlabel("Selfing rate")
    ax.set_ylabel("Median delta log-likelihood" if delta else "Median log-likelihood")
    if not summary.empty:
        ax.legend(title="Group")
    return save_or_return(fig, out)


def plot_best_s_distribution(
    fit_summary: pd.DataFrame | str | Path,
    *,
    out: str | Path | None = None,
):
    """Plot draw-level best selfing-rate estimates by group."""
    plt = require_pyplot()
    frame = dataframe_or_csv(fit_summary)
    _require_columns(frame, {"group", "best_s"}, "fit summary")
    frame = frame.dropna(subset=["best_s"]).copy()

    fig, ax = plt.subplots(figsize=(7, 4))
    groups = sorted(frame["group"].astype(str).unique())
    if groups:
        values = [frame.loc[frame["group"].astype(str) == group, "best_s"].to_numpy() for group in groups]
        ax.boxplot(values, showfliers=False)
        ax.set_xticklabels(groups)
        for position, group_values in enumerate(values, start=1):
            if len(group_values) == 0:
                continue
            jitter = np.linspace(-0.08, 0.08, len(group_values)) if len(group_values) > 1 else [0.0]
            ax.scatter(position + np.asarray(jitter), group_values, s=18, alpha=0.7)
    ax.set_xlabel("Group")
    ax.set_ylabel("Best selfing rate")
    ax.set_ylim(-0.02, 1.02)
    return save_or_return(fig, out)


def plot_representative_observed_vs_fitted(
    observed_vs_fitted: pd.DataFrame | str | Path,
    *,
    out: str | Path | None = None,
):
    """Plot observed and fitted DGS counts for representative empirical draws."""
    plt = require_pyplot()
    frame = dataframe_or_csv(observed_vs_fitted)
    _require_columns(
        frame,
        {"group", "n0", "n1", "n2", "observed_count", "fitted_expected_count"},
        "representative observed-vs-fitted",
    )
    frame = frame.copy()
    frame["cell"] = _cell_labels(frame)

    groups = sorted(frame["group"].astype(str).unique())
    n_panels = max(len(groups), 1)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(max(7, 0.5 * max(len(frame), 1)), 3.5 * n_panels),
        squeeze=False,
    )
    for ax, group in zip(axes[:, 0], groups):
        group_frame = frame[frame["group"].astype(str) == group].sort_values(["n0", "n1", "n2"])
        positions = np.arange(len(group_frame))
        ax.bar(positions - 0.18, group_frame["observed_count"], width=0.36, label="Observed")
        ax.bar(
            positions + 0.18,
            group_frame["fitted_expected_count"],
            width=0.36,
            label="Fitted",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(group_frame["cell"], rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(str(group))
        ax.legend()
    if not groups:
        axes[0, 0].set_xlabel("DGS cell")
        axes[0, 0].set_ylabel("Count")
    fig.tight_layout()
    return save_or_return(fig, out)
