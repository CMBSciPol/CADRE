"""Optional-dependency guard decorators for cadre."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

Param = ParamSpec("Param")
ReturnType = TypeVar("ReturnType")

try:
    import jaxopt  # noqa: F401
except ImportError:
    pass


def requires_scipy(func: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
    """Decorator that raises ImportError if jaxopt is not available.

    Apply to any function that requires the optional ``cadre[scipy]`` extras
    (``jaxopt``, ``cobyqa``, ``scipy``).
    """
    try:
        import jaxopt  # noqa: F401

        return func
    except ImportError:
        pass

    @wraps(func)
    def deferred_func(*args: Param.args, **kwargs: Param.kwargs) -> ReturnType:
        msg = "Missing optional scipy solvers. " "Install with: pip install cadre[scipy]"
        raise ImportError(msg)

    return deferred_func  # type: ignore[return-value]
