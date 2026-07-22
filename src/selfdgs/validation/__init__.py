"""Optional method-validation experiments and result consolidation.

Routine inference from observed data lives in the root package and does not
depend on this subpackage.
"""

from selfdgs.validation.experiments import (
    LocusResult,
    ReplicateResult,
    ValidationExperimentConfig,
    run_validation_experiment,
    summarize_existing_vcfs,
)
from selfdgs.validation.consolidation import (
    discover_validation_runs,
    load_likelihood_collection,
    load_validation_collection,
    summarize_estimator_recovery,
)

__all__ = [
    "LocusResult",
    "ReplicateResult",
    "ValidationExperimentConfig",
    "discover_validation_runs",
    "load_likelihood_collection",
    "load_validation_collection",
    "run_validation_experiment",
    "summarize_estimator_recovery",
    "summarize_existing_vcfs",
]
