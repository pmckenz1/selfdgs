"""DGS spectrum plotting helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from selfdgs.spectrum import DGSConfig, dgs_to_dataframe
from selfdgs.plotting._common import require_pyplot, save_or_return


def _dgs_frame(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, Mapping):
        frame = dgs_to_dataframe(data)
    else:
        frame = pd.read_csv(data)

    required = {"n0", "n1", "n2", "count"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"DGS data are missing columns: {sorted(missing)}.")
    frame = frame.copy()
    frame["cell"] = frame.apply(
        lambda row: f"({int(row.n0)},{int(row.n1)},{int(row.n2)})",
        axis=1,
    )
    return frame.sort_values(["n0", "n1", "n2"])


def plot_dgs_counts(
    counts: Mapping[DGSConfig, int] | pd.DataFrame | str | Path,
    *,
    out: str | Path | None = None,
    ax=None,
    title: str | None = None,
):
    """Plot observed DGS counts as a bar chart."""
    plt = require_pyplot()
    frame = _dgs_frame(counts)
    if ax is None:
        width = max(6, min(14, 0.45 * max(len(frame), 1)))
        fig, ax = plt.subplots(figsize=(width, 4))
    else:
        fig = ax.figure

    ax.bar(frame["cell"], frame["count"], color="#3b6ea8")
    ax.set_xlabel("DGS cell (n0,n1,n2)")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    return save_or_return(fig, out)

