"""Structured fit results for selfdgs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class LikelihoodPoint:
    """One evaluated selfing-rate point on a likelihood curve."""

    s: float
    loglik: float
    delta_loglik: float = 0.0


@dataclass(frozen=True)
class FitResult:
    """Result of fitting a selfing-rate model to observed DGS counts."""

    best_s: float
    best_loglik: float
    n_sites: int
    n_diploids: int
    method: str
    likelihood_curve: tuple[LikelihoodPoint, ...] = field(default_factory=tuple)
    support_interval: tuple[float, float] | None = None
    boundary_optimum: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)
    n_input_sites: int | None = None
    n_excluded_sites: int = 0
    n_excluded_cells: int = 0

    def likelihood_dataframe(self) -> pd.DataFrame:
        """Return the likelihood curve as a DataFrame."""
        return pd.DataFrame(
            [asdict(point) for point in self.likelihood_curve],
            columns=["s", "loglik", "delta_loglik"],
        )

    def summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable fit summary."""
        return {
            "best_s": self.best_s,
            "best_loglik": self.best_loglik,
            "n_sites": self.n_sites,
            "n_diploids": self.n_diploids,
            "method": self.method,
            "support_interval": self.support_interval,
            "boundary_optimum": self.boundary_optimum,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
            "n_input_sites": int(
                self.n_sites if self.n_input_sites is None else self.n_input_sites
            ),
            "n_excluded_sites": int(self.n_excluded_sites),
            "n_excluded_cells": int(self.n_excluded_cells),
        }

    def to_json(self, path: str | Path) -> None:
        """Write the fit summary and likelihood curve to JSON."""
        payload = self.summary_dict()
        payload["likelihood_curve"] = [
            {
                "s": point.s,
                "loglik": point.loglik if math.isfinite(point.loglik) else None,
                "delta_loglik": (
                    point.delta_loglik if math.isfinite(point.delta_loglik) else None
                ),
            }
            for point in self.likelihood_curve
        ]
        Path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    def to_csv(self, path: str | Path) -> None:
        """Write the likelihood curve to CSV."""
        self.likelihood_dataframe().to_csv(path, index=False)


def fit_result_from_json(path: str | Path) -> FitResult:
    """Read a ``FitResult`` previously written with ``FitResult.to_json``."""
    payload = json.loads(Path(path).read_text())
    curve = tuple(
        LikelihoodPoint(
            s=float(point["s"]),
            loglik=(
                float(point["loglik"])
                if point.get("loglik") is not None
                else float("-inf")
            ),
            delta_loglik=(
                float(point["delta_loglik"])
                if point.get("delta_loglik") is not None
                else float("-inf")
            ),
        )
        for point in payload.pop("likelihood_curve", [])
    )
    support_interval = payload.get("support_interval")
    if support_interval is not None:
        support_interval = (float(support_interval[0]), float(support_interval[1]))
    return FitResult(
        best_s=float(payload["best_s"]),
        best_loglik=float(payload["best_loglik"]),
        n_sites=int(payload["n_sites"]),
        n_diploids=int(payload["n_diploids"]),
        method=str(payload["method"]),
        likelihood_curve=curve,
        support_interval=support_interval,
        boundary_optimum=bool(payload.get("boundary_optimum", False)),
        warnings=tuple(payload.get("warnings", [])),
        metadata=dict(payload.get("metadata", {})),
        n_input_sites=(
            int(payload["n_input_sites"])
            if payload.get("n_input_sites") is not None
            else None
        ),
        n_excluded_sites=int(payload.get("n_excluded_sites", 0)),
        n_excluded_cells=int(payload.get("n_excluded_cells", 0)),
    )
