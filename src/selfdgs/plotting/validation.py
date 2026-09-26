"""Validation diagnostic plotting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from selfdgs.plotting._common import dataframe_or_csv, require_pyplot, save_or_return
from selfdgs.plotting.likelihood import _require_identical_s_grids


def plot_replicate_estimates(
    summary: pd.DataFrame | str | Path,
    *,
    out: str | Path | None = None,
    true_s: float | None = None,
    bins: int | list[float] = 20,
):
    """Plot a histogram of replicate best selfing-rate estimates."""
    plt = require_pyplot()
    frame = dataframe_or_csv(summary)
    if "best_s" not in frame.columns:
        raise ValueError("replicate summary is missing required column: best_s.")
    frame = frame.dropna(subset=["best_s"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(frame["best_s"], bins=bins, edgecolor="black", color="#6aa36f")
    if true_s is not None:
        ax.axvline(true_s, color="black", linestyle="--", linewidth=1, label="true s")
        ax.legend()
    ax.set_xlabel("Estimated selfing rate")
    ax.set_ylabel("Replicates")
    return save_or_return(fig, out)


def plot_cumulative_convergence(
    likelihoods: pd.DataFrame | str | Path,
    *,
    out: str | Path | None = None,
    true_s: float | None = None,
):
    """Plot cumulative estimates across independent replicates at one true rate.

    Replicates must not be nested samples or paired analyses of one population.
    """
    plt = require_pyplot()
    frame = dataframe_or_csv(likelihoods)
    required = {"rep", "s", "loglik"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"cumulative likelihood data are missing columns: {sorted(missing)}.")
    _require_identical_s_grids(frame, curve_columns=("rep",))

    rows = []
    reps = sorted(frame["rep"].unique())
    for k, rep in enumerate(reps, start=1):
        sub = frame[frame["rep"].isin(reps[:k])]
        cumulative = sub.groupby("s", as_index=False)["loglik"].sum()
        best = cumulative.loc[cumulative["loglik"].idxmax()]
        rows.append({"n_reps": k, "max_rep": rep, "best_s": best["s"], "loglik": best["loglik"]})
    cumulative_frame = pd.DataFrame(
        rows,
        columns=["n_reps", "max_rep", "best_s", "loglik"],
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cumulative_frame["n_reps"], cumulative_frame["best_s"], marker="o", linewidth=1.5)
    if true_s is not None:
        ax.axhline(true_s, color="black", linestyle="--", linewidth=1, label="true s")
        ax.legend()
    ax.set_xlabel("Number of replicates")
    ax.set_ylabel("Cumulative best selfing rate")
    return save_or_return(fig, out)
