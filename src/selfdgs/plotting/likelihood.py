"""Likelihood plotting helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from selfdgs.results import FitResult, fit_result_from_json
from selfdgs.plotting._common import require_pyplot, save_or_return


def _require_identical_s_grids(
    frame: pd.DataFrame,
    *,
    curve_columns: tuple[str, ...] | None = None,
) -> None:
    """Reject likelihood collections whose component curves use different grids."""
    if curve_columns is None:
        candidates = (
            ("run_id", "rep"),
            ("group", "draw_id"),
            ("rep",),
            ("draw_id",),
            ("curve",),
        )
        curve_columns = next(
            (columns for columns in candidates if all(c in frame.columns for c in columns)),
            (),
        )
    if curve_columns:
        key_columns = [*curve_columns, "s"]
        if frame.duplicated(key_columns, keep=False).any():
            raise ValueError(
                "likelihood curves must contain exactly one row per curve and "
                "selfing-rate value."
            )
        grids = [
            tuple(sorted(group["s"].astype(float).unique()))
            for _, group in frame.groupby(list(curve_columns), dropna=False, sort=False)
        ]
        if grids and any(grid != grids[0] for grid in grids[1:]):
            raise ValueError("likelihood curves must use identical selfing-rate grids.")
        return
    if frame["s"].duplicated().any():
        raise ValueError(
            "likelihood data with repeated selfing-rate values must include "
            "curve-identifying columns."
        )


def _likelihood_frame(data) -> pd.DataFrame:
    if isinstance(data, FitResult):
        frame = data.likelihood_dataframe()
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        path = Path(data)
        if path.suffix.lower() == ".json":
            frame = fit_result_from_json(path).likelihood_dataframe()
        else:
            frame = pd.read_csv(path)

    required = {"s", "loglik"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"likelihood data are missing columns: {sorted(missing)}.")
    if "delta_loglik" not in frame.columns:
        frame["delta_loglik"] = frame["loglik"] - frame["loglik"].max()
    return frame.sort_values("s")


def plot_likelihood_curve(
    data,
    *,
    out: str | Path | None = None,
    delta: bool = True,
    true_s: float | None = None,
    ax=None,
    label: str | None = None,
):
    """Plot one likelihood curve from a ``FitResult``, CSV/JSON path, or DataFrame."""
    plt = require_pyplot()
    frame = _likelihood_frame(data)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    y_column = "delta_loglik" if delta else "loglik"
    ax.plot(frame["s"], frame[y_column], marker="o", linewidth=1.5, label=label)
    if true_s is not None:
        ax.axvline(true_s, color="black", linestyle="--", linewidth=1, label="true s")
    ax.set_xlabel("Selfing rate")
    ax.set_ylabel("Delta log-likelihood" if delta else "Log-likelihood")
    if label is not None or true_s is not None:
        ax.legend()
    return save_or_return(fig, out)


def plot_likelihood_curves(
    curves: Mapping[str, object] | Iterable[object],
    *,
    out: str | Path | None = None,
    delta: bool = True,
    true_s: float | None = None,
):
    """Plot multiple likelihood curves on one axis."""
    plt = require_pyplot()
    fig, ax = plt.subplots(figsize=(7, 4))

    if isinstance(curves, Mapping):
        items = curves.items()
    else:
        items = ((None, curve) for curve in curves)

    for label, curve in items:
        plot_likelihood_curve(curve, delta=delta, true_s=None, ax=ax, label=label)
    if true_s is not None:
        ax.axvline(true_s, color="black", linestyle="--", linewidth=1, label="true s")
    if isinstance(curves, Mapping) or true_s is not None:
        ax.legend()
    return save_or_return(fig, out)


def plot_joint_likelihood(
    likelihoods: pd.DataFrame | str | Path,
    *,
    out: str | Path | None = None,
    true_s: float | None = None,
):
    """Plot a joint likelihood curve by summing log-likelihoods by ``s``."""
    plt = require_pyplot()
    frame = pd.read_csv(likelihoods) if not isinstance(likelihoods, pd.DataFrame) else likelihoods.copy()
    required = {"s", "loglik"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"joint likelihood data are missing columns: {sorted(missing)}.")
    _require_identical_s_grids(frame)

    joint = frame.groupby("s", as_index=False)["loglik"].sum()
    joint["delta_loglik"] = joint["loglik"] - joint["loglik"].max()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(joint["s"], joint["delta_loglik"], marker="o", linewidth=1.5)
    if true_s is not None:
        ax.axvline(true_s, color="black", linestyle="--", linewidth=1, label="true s")
        ax.legend()
    ax.set_xlabel("Selfing rate")
    ax.set_ylabel("Joint delta log-likelihood")
    return save_or_return(fig, out)
