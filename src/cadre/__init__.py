"""CADRE — Constraint-Aware Descent Routine Executor.

A JAX-native optimization library providing:
- Active-set constrained optimization (ADABK and LBFGSK families)
- L-BFGS with zoom/backtracking linesearch (off-the-shelf)
- Unified interface to optax, optimistix, and scipy solvers
- Parameter conditioning, box projection, and Gaussian-prior utilities
"""

from importlib import metadata

from .active_set import ActiveSetMinimiser, ActiveSetState, active_set
from .adabk import make_adabk_solver
from .constraints import BoxConstraint, Constraint, GaussianConstraint, validate_constraint
from .lbfgsk import make_lbfgsk_solver
from .minimize import ScipyMinimizeState, UnifiedState, minimize, scipy_minimize
from .solvers import (
    SELFCONDITIONED_SOLVERS,
    SOLVER_NAMES,
    apply_projection,
    get_solver,
    lbfgs_backtrack,
    lbfgs_zoom,
)
from .utils import condition

__all__ = [
    # Core optimizer
    "active_set",
    "ActiveSetState",
    "ActiveSetMinimiser",
    # ADABK / LBFGSK families
    "make_adabk_solver",
    "make_lbfgsk_solver",
    # Unified interface
    "minimize",
    "scipy_minimize",
    "ScipyMinimizeState",
    "UnifiedState",
    # Solver factory
    "get_solver",
    "SOLVER_NAMES",
    "SELFCONDITIONED_SOLVERS",
    # L-BFGS variants
    "lbfgs_zoom",
    "lbfgs_backtrack",
    # Constraints
    "BoxConstraint",
    "GaussianConstraint",
    "Constraint",
    "validate_constraint",
    # Utilities
    "apply_projection",
    "condition",
]


def __getattr__(name: str) -> str:
    """Expose package metadata attributes lazily."""
    if name == "__version__":
        try:
            return metadata.version("jax-cadre")
        except metadata.PackageNotFoundError:
            return "unknown"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
