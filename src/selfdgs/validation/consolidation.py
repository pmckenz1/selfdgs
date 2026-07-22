"""Consolidation helpers for saved validation simulation outputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

RUN_NAME_RE = re.compile(
    r"(?:^|_)ne(?P<ne>\d+)(?:_|-)?self(?P<true_s>[0-9]+(?:\.[0-9]+)?)"
)
REPLICATE_GLOB = "rep[0-9]*"
VALIDATION_COLLECTION_COLUMNS = [
    "run_id",
    "run_path",
    "base_dir",
    "rep",
    "ne",
    "true_s",
    "n_diploids",
    "best_s",
    "best_loglik",
    "n_sites",
    "n_loci",
    "boundary_optimum",
    "observed_dgs_path",
    "observed_polymorphic_dgs_path",
    "likelihood_path",
]


def _as_base_dirs(base_dirs: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(base_dirs, (str, Path)):
        return [Path(base_dirs)]
    return [Path(path) for path in base_dirs]


def _parse_run_name(path: Path) -> dict[str, object]:
    match = RUN_NAME_RE.search(path.name)
    if not match:
        return {}
    return {
        "ne": int(match.group("ne")),
        "true_s": float(match.group("true_s")),
    }


def _read_first_row(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, nrows=1)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _read_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text()))


def _first_replicate_file(run_dir: Path, filename: str) -> Path | None:
    paths = sorted(run_dir.glob(f"{REPLICATE_GLOB}/{filename}"))
    return paths[0] if paths else None


def _replicate_dir(run_dir: Path, rep: object) -> Path | None:
    """Resolve canonical and legacy replicate directory names by replicate ID."""
    try:
        rep_id = int(rep)
    except (TypeError, ValueError):
        return None
    canonical = run_dir / f"rep{rep_id:03d}"
    if canonical.is_dir():
        return canonical
    for path in sorted(run_dir.glob(REPLICATE_GLOB)):
        match = re.fullmatch(r"rep(\d+).*", path.name)
        if path.is_dir() and match and int(match.group(1)) == rep_id:
            return path
    return None


def _candidate_run_dirs(base_dir: Path, pattern: str) -> list[Path]:
    candidates = [path for path in base_dir.glob(pattern) if path.is_dir()]
    if any(
        (base_dir / name).exists()
        for name in (
            "validation_summary.csv",
            "validation_config.json",
            "fit_summary.csv",
        )
    ):
        candidates.append(base_dir)
    return sorted(set(candidates), key=lambda path: str(path))


def discover_validation_runs(
    base_dirs: str | Path | Iterable[str | Path],
    *,
    pattern: str = "ne*_self*",
) -> pd.DataFrame:
    """Discover saved validation run directories.

    Directories are considered runs when they match `pattern`, or when the base
    directory itself contains `validation_summary.csv` or `fit_summary.csv`.
    Directory names such as `ne100_self0.500` are parsed into `ne` and `true_s`
    when possible.
    """
    rows: list[dict[str, object]] = []
    for base_dir in _as_base_dirs(base_dirs):
        for run_dir in _candidate_run_dirs(base_dir, pattern):
            parsed = _parse_run_name(run_dir)
            summary = _read_first_row(run_dir / "validation_summary.csv")
            config = _read_config(run_dir / "validation_config.json")
            fit_path = run_dir / "fit_summary.csv"
            if not fit_path.exists():
                fit_path = _first_replicate_file(run_dir, "fit_summary.csv") or fit_path
            fit = _read_first_row(fit_path)
            row = {
                "run_id": run_dir.name,
                "path": str(run_dir),
                "base_dir": str(base_dir),
                "has_validation_summary": (run_dir / "validation_summary.csv").exists(),
                "has_fit_summary": (run_dir / "fit_summary.csv").exists()
                or _first_replicate_file(run_dir, "fit_summary.csv") is not None,
                "has_likelihood": (run_dir / "likelihood.csv").exists()
                or any(run_dir.glob(f"{REPLICATE_GLOB}/likelihood.csv")),
                "has_observed_dgs": (run_dir / "observed_dgs.csv").exists()
                or _first_replicate_file(run_dir, "observed_dgs.csv") is not None,
                "has_observed_polymorphic_dgs": (
                    run_dir / "observed_polymorphic_dgs.csv"
                ).exists()
                or _first_replicate_file(
                    run_dir,
                    "observed_polymorphic_dgs.csv",
                )
                is not None,
            }
            row.update(parsed)
            for column in ("ne", "true_s"):
                if column not in row and column in summary:
                    row[column] = summary[column]
                elif column not in row and column in config:
                    row[column] = config[column]
            if "n_diploids" in fit:
                row["n_diploids"] = fit["n_diploids"]
            elif "n_sample" in config:
                row["n_diploids"] = config["n_sample"]
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "path",
                "base_dir",
                "has_validation_summary",
                "has_fit_summary",
                "has_likelihood",
                "has_observed_dgs",
                "has_observed_polymorphic_dgs",
            ]
        )
    return pd.DataFrame(rows).sort_values(["base_dir", "run_id"]).reset_index(drop=True)


def _runs_frame(runs: pd.DataFrame | Sequence[Mapping[str, object]] | Sequence[str | Path]) -> pd.DataFrame:
    if isinstance(runs, pd.DataFrame):
        return runs.copy()
    if not runs:
        return pd.DataFrame(columns=["path"])
    first = runs[0]  # type: ignore[index]
    if isinstance(first, Mapping):
        return pd.DataFrame(list(runs))  # type: ignore[arg-type]
    return pd.DataFrame({"path": [str(path) for path in runs]})  # type: ignore[arg-type]


def _annotate_run_rows(frame: pd.DataFrame, run: Mapping[str, object], run_dir: Path) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "run_id", str(run.get("run_id", run_dir.name)))
    out.insert(1, "run_path", str(run_dir))
    for column in ("base_dir", "ne", "true_s", "n_diploids"):
        if column in run and column not in out.columns:
            out[column] = run[column]
    if "n_diploids" not in out.columns:
        fit_path = run_dir / "fit_summary.csv"
        if not fit_path.exists():
            fit_path = _first_replicate_file(run_dir, "fit_summary.csv") or fit_path
        fit = _read_first_row(fit_path)
        if "n_diploids" in fit:
            out["n_diploids"] = fit["n_diploids"]
    for file_column, filename in (
        ("observed_dgs_path", "observed_dgs.csv"),
        ("observed_polymorphic_dgs_path", "observed_polymorphic_dgs.csv"),
        ("likelihood_path", "likelihood.csv"),
    ):
        if file_column in out.columns:
            continue
        paths: list[object] = []
        for row in out.itertuples(index=False):
            path = run_dir / filename
            if not path.exists() and hasattr(row, "rep"):
                replicate_dir = _replicate_dir(run_dir, row.rep)
                if replicate_dir is not None:
                    path = replicate_dir / filename
            paths.append(str(path) if path.exists() else pd.NA)
        if any(not pd.isna(path) for path in paths):
            out[file_column] = paths
    return out


def load_validation_collection(
    runs: pd.DataFrame | Sequence[Mapping[str, object]] | Sequence[str | Path],
) -> pd.DataFrame:
    """Load validation summary rows from discovered run directories."""
    run_frame = _runs_frame(runs)
    frames: list[pd.DataFrame] = []
    for run in run_frame.to_dict("records"):
        run_dir = Path(str(run["path"]))
        if (run_dir / "validation_summary.csv").exists():
            frame = pd.read_csv(run_dir / "validation_summary.csv")
        elif (run_dir / "fit_summary.csv").exists():
            frame = pd.read_csv(run_dir / "fit_summary.csv")
        else:
            replicate_frames = []
            for fit_path in sorted(run_dir.glob(f"{REPLICATE_GLOB}/fit_summary.csv")):
                replicate = pd.read_csv(fit_path)
                if "rep" not in replicate.columns:
                    match = re.search(r"rep(\d+)", fit_path.parent.name)
                    if match:
                        replicate["rep"] = int(match.group(1))
                replicate_frames.append(replicate)
            if not replicate_frames:
                continue
            frame = pd.concat(replicate_frames, ignore_index=True)
        frames.append(_annotate_run_rows(frame, run, run_dir))
    if not frames:
        return pd.DataFrame(columns=VALIDATION_COLLECTION_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_likelihood_collection(
    runs: pd.DataFrame | Sequence[Mapping[str, object]] | Sequence[str | Path],
    *,
    require: bool = False,
) -> pd.DataFrame:
    """Load and annotate available likelihood curves, skipping missing files by default."""
    run_frame = _runs_frame(runs)
    frames: list[pd.DataFrame] = []
    for run in run_frame.to_dict("records"):
        run_dir = Path(str(run["path"]))
        likelihood_paths = [run_dir / "likelihood.csv"] if (run_dir / "likelihood.csv").exists() else []
        likelihood_paths.extend(
            sorted(run_dir.glob(f"{REPLICATE_GLOB}/likelihood.csv"))
        )
        if not likelihood_paths:
            if require:
                raise FileNotFoundError(run_dir / "likelihood.csv")
            continue
        for likelihood_path in likelihood_paths:
            frame = pd.read_csv(likelihood_path)
            if "rep" not in frame.columns:
                match = re.search(r"rep(\d+)", likelihood_path.parent.name)
                if match:
                    frame["rep"] = int(match.group(1))
            annotated = _annotate_run_rows(frame, run, run_dir)
            annotated["likelihood_path"] = str(likelihood_path)
            frames.append(annotated)
    if not frames:
        return pd.DataFrame(columns=["run_id", "run_path", "s", "loglik", "delta_loglik"])
    return pd.concat(frames, ignore_index=True)


def summarize_estimator_recovery(
    validation_table: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("ne", "true_s"),
    estimate_column: str = "best_s",
    true_column: str = "true_s",
) -> pd.DataFrame:
    """Summarize estimator recovery bias, error, and RMSE by simulation setting."""
    missing = {estimate_column, true_column}.difference(validation_table.columns)
    if missing:
        raise ValueError(f"validation_table is missing required columns: {sorted(missing)}")
    group_cols = [column for column in group_columns if column in validation_table.columns]
    frame = validation_table.dropna(subset=[estimate_column, true_column]).copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[*group_cols, "n_runs", "mean_estimate", "median_estimate", "bias", "mae", "rmse"]
        )
    frame["_error"] = frame[estimate_column].astype(float) - frame[true_column].astype(float)
    grouped = frame.groupby(group_cols, dropna=False) if group_cols else [((), frame)]
    rows = []
    for key, group in grouped:
        row = {}
        if group_cols:
            key_tuple = key if isinstance(key, tuple) else (key,)
            row.update(dict(zip(group_cols, key_tuple)))
        errors = group["_error"].astype(float)
        estimates = group[estimate_column].astype(float)
        row.update(
            {
                "n_runs": int(len(group)),
                "mean_estimate": float(estimates.mean()),
                "median_estimate": float(estimates.median()),
                "bias": float(errors.mean()),
                "mae": float(errors.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
