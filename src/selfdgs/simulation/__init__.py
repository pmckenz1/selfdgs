"""Optional SLiM execution primitives for method-validation workflows.

Routine inference from observed data does not depend on this subpackage.
"""

from selfdgs.simulation.slim import (
    SLiMUnavailableError,
    SlimRunResult,
    SlimSimulationConfig,
    build_slim_command,
    default_slim_script_path,
    run_slim_simulation,
    slim_available,
)

__all__ = [
    "SLiMUnavailableError",
    "SlimRunResult",
    "SlimSimulationConfig",
    "build_slim_command",
    "default_slim_script_path",
    "run_slim_simulation",
    "slim_available",
]
