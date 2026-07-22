"""Failure-safe installation of staged workflow outputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _move_path(source: Path, destination: Path) -> None:
    """Atomically rename one output path on the shared output filesystem."""
    source.replace(destination)


def install_staged_outputs(
    staged_outdir: str | Path,
    outdir: str | Path,
    *,
    owned_names: Iterable[str],
) -> None:
    """Install staged direct children and restore previous outputs on failure.

    Only names in ``owned_names`` may be installed or removed. Other files in
    an existing output directory are left untouched.
    """
    staged_outdir = Path(staged_outdir)
    outdir = Path(outdir)
    owned = set(owned_names)
    if any(not name or Path(name).name != name for name in owned):
        raise ValueError("owned output names must be direct path components")

    staged_paths = sorted(staged_outdir.iterdir(), key=lambda path: path.name)
    unexpected = sorted(path.name for path in staged_paths if path.name not in owned)
    if unexpected:
        raise ValueError(f"staged output contains unowned paths: {unexpected}")

    if outdir.exists() and not outdir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {outdir}")
    if outdir.is_symlink():
        raise ValueError(f"Output directory cannot be a symbolic link: {outdir}")

    outdir.parent.mkdir(parents=True, exist_ok=True)
    created_outdir = not outdir.exists()
    outdir.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=f".{outdir.name}.backup.", dir=outdir.parent) as tempdir:
        backup = Path(tempdir)
        backed_up: list[str] = []
        installed: list[str] = []
        try:
            for name in sorted(owned):
                current = outdir / name
                if current.exists() or current.is_symlink():
                    _move_path(current, backup / name)
                    backed_up.append(name)
            for staged in staged_paths:
                _move_path(staged, outdir / staged.name)
                installed.append(staged.name)
        except Exception:
            for name in reversed(installed):
                _remove_path(outdir / name)
            for name in backed_up:
                _move_path(backup / name, outdir / name)
            if created_outdir:
                try:
                    outdir.rmdir()
                except OSError:
                    pass
            raise
