"""Shared plotting utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def dataframe_or_csv(data) -> pd.DataFrame:
    """Return a defensive DataFrame copy or load one from CSV."""
    return data.copy() if isinstance(data, pd.DataFrame) else pd.read_csv(data)


def require_pyplot():
    """Import matplotlib lazily and return ``matplotlib.pyplot``."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "selfdgs plotting requires matplotlib; reinstall selfdgs to restore "
            "its required dependencies."
        ) from exc
    return plt


def save_or_return(fig, out: str | Path | None):
    """Save a figure when ``out`` is provided, otherwise return it."""
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    return fig
