"""CADRE — Constraint-Aware Descent Routine Executor.

A JAX-native optimization library providing:
- Active-set constrained optimization (ADABK family)
- L-BFGS with zoom/backtracking linesearch
- Unified interface to optax, optimistix, and scipy solvers
- Parameter conditioning and box projection utilities
"""

from importlib import metadata

from .active_set import ActiveSetState, active_set
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
