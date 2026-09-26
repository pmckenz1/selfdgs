"""Optional SLiM simulation support."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib import resources
import math
from numbers import Integral, Real
from pathlib import Path
import re
import shutil
import subprocess
from typing import Mapping


class SLiMUnavailableError(RuntimeError):
    """Raised when a SLiM run is requested but SLiM is not available."""


def default_slim_script_path(sampling_design: str = "fixed_individuals") -> Path:
    """Return the packaged script for the requested sampling design.

    The package is installed from ordinary wheels as unpacked files, so this
    path is suitable for passing directly to the `slim` executable.
    """
    scripts = {
        "independent_populations": "equilibrium_selfing.slim",
        "fixed_individuals": "fixed_individuals.slim",
    }
    if sampling_design not in scripts:
        raise ValueError(f"Unknown sampling design: {sampling_design!r}")
    return Path(resources.files("selfdgs.resources") / scripts[sampling_design])


@dataclass(frozen=True)
class SlimSimulationConfig:
    """Configuration for one SLiM VCF simulation."""

    slim_script: str | Path
    vcf_path: str | Path
    ne: int | None = field(default=None, repr=False)
    selfing_rate: float = 0.5
    mu: float = 1e-7
    chrom_length: int = 10_000
    recomb_rate: float = 5e-8
    n_sample: int = 4
    burn_mult: int = 10
    seed: int = 1
    slim_executable: str = "slim"
    extra_defines: Mapping[str, object] = field(default_factory=dict)

    census_size: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.census_size is not None and self.ne is not None and self.census_size != self.ne:
            raise ValueError("census_size and ne must agree when both are supplied.")
        size = self.census_size if self.census_size is not None else self.ne
        size = 500 if size is None else size
        object.__setattr__(self, "census_size", size)
        object.__setattr__(self, "ne", size)
        positive_integers = {
            "census_size": self.census_size,
            "chrom_length": self.chrom_length,
            "n_sample": self.n_sample,
            "seed": self.seed,
        }
        invalid = {
            name: value
            for name, value in positive_integers.items()
            if isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < 1
        }
        if invalid:
            raise ValueError(f"SLiM integer parameters must be positive integers: {invalid}")
        if (
            isinstance(self.burn_mult, bool)
            or not isinstance(self.burn_mult, Integral)
            or self.burn_mult < 0
        ):
            raise ValueError("burn_mult must be a non-negative integer.")
        if (
            isinstance(self.selfing_rate, bool)
            or not isinstance(self.selfing_rate, Real)
            or not math.isfinite(self.selfing_rate)
            or not 0.0 <= self.selfing_rate <= 1.0
        ):
            raise ValueError("selfing_rate must be finite and in [0, 1].")
        for name, value in (("mu", self.mu), ("recomb_rate", self.recomb_rate)):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative.")
        if not isinstance(self.slim_executable, str) or not self.slim_executable.strip():
            raise ValueError("slim_executable must be a non-empty string.")
        if not isinstance(self.extra_defines, Mapping):
            raise ValueError("extra_defines must be a mapping.")
        invalid_names = sorted(
            str(name)
            for name in self.extra_defines
            if not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
        )
        if invalid_names:
            raise ValueError(
                f"extra_defines contains invalid Eidos identifiers: {invalid_names}"
            )
        for value in self.extra_defines.values():
            _format_define_value(value)

    def metadata(self) -> dict[str, object]:
        """Return JSON/CSV friendly simulation metadata."""
        data = asdict(self)
        data.pop("ne")
        data["slim_script"] = str(self.slim_script)
        data["vcf_path"] = str(self.vcf_path)
        data["extra_defines"] = dict(self.extra_defines)
        return data


@dataclass(frozen=True)
class SlimRunResult:
    """Result from one SLiM subprocess execution."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    vcf_path: str
    metadata: dict[str, object]


def _format_define_value(value: object) -> str:
    if value is None:
        raise ValueError("SLiM define values cannot be None.")
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, (Path, str)):
        text = value.as_posix() if isinstance(value, Path) else value
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, Real):
        if not math.isfinite(value):
            raise ValueError("SLiM numeric define values must be finite.")
        return str(value)
    raise ValueError(
        "SLiM define values must be strings, paths, booleans, or finite numbers."
    )


@lru_cache(maxsize=None)
def _slim_version(slim_executable: str) -> str | None:
    """Return SLiM's reported version without making version lookup fatal."""
    try:
        completed = subprocess.run(
            [slim_executable, "-v"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = completed.stdout.strip() or completed.stderr.strip()
    return output.splitlines()[0] if output else None


def build_slim_command(config: SlimSimulationConfig) -> list[str]:
    """Build the SLiM command for a simulation without executing it."""
    defines: dict[str, object] = {
        "ne": config.ne,
        "selfing_rate": config.selfing_rate,
        "mu": config.mu,
        "chrom_length": config.chrom_length,
        "recomb_rate": config.recomb_rate,
        "n_sample": config.n_sample,
        "burn_mult": config.burn_mult,
        "seed": config.seed,
        "vcf_path": Path(config.vcf_path),
    }
    extra_defines = dict(config.extra_defines)
    reserved = sorted(defines.keys() & extra_defines.keys())
    if reserved:
        raise ValueError(
            "extra_defines cannot override dedicated simulation settings: "
            f"{reserved}"
        )
    defines.update(extra_defines)

    command = [config.slim_executable]
    for key, value in defines.items():
        command.extend(["-d", f"{key}={_format_define_value(value)}"])
    command.append(str(config.slim_script))
    return command


def slim_available(slim_executable: str = "slim") -> bool:
    """Return true if the SLiM executable can be found on ``PATH``."""
    return shutil.which(slim_executable) is not None


def run_slim_simulation(
    config: SlimSimulationConfig,
    *,
    check: bool = True,
    timeout: float | None = None,
) -> SlimRunResult:
    """Run one SLiM simulation and return captured subprocess output."""
    if not slim_available(config.slim_executable):
        raise SLiMUnavailableError(
            f"SLiM executable {config.slim_executable!r} was not found on PATH."
        )

    Path(config.vcf_path).parent.mkdir(parents=True, exist_ok=True)
    command = build_slim_command(config)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            "SLiM failed with return code "
            f"{completed.returncode}:\nCOMMAND: {' '.join(command)}\n\n"
            f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )
    metadata = config.metadata()
    metadata["slim_version"] = _slim_version(config.slim_executable)
    return SlimRunResult(
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        vcf_path=str(config.vcf_path),
        metadata=metadata,
    )
