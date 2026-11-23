from __future__ import annotations

from typing import Any, Literal, NamedTuple, TypeAlias, Union

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
import optimistix as optx
from jaxtyping import Array, Float, Int, PyTree
from optax._src import combine, transform
from optax._src import linesearch as _linesearch

from .adatopk import AdaTopK

# =============================================================================
# OFF-THE-SHELF L-BFGS SOLVERS
# =============================================================================

Solver: TypeAlias = Union[optx.BestSoFarMinimiser, AdaTopK, str]


def lbfgs_zoom(
    learning_rate: optax.ScalarOrSchedule | None = None,
    memory_size: int = 10,
    scale_init_precond: bool = False,
    max_linesearch_steps: int = 200,
    initial_guess_strategy: str = "one",
    slope_rtol: float = 1e-4,
    curv_rtol: float = 0.9,
    verbose: bool = False,
    lower: PyTree[Float[Array, " P"]] | None = None,
    upper: PyTree[Float[Array, " P"]] | None = None,
) -> optax.GradientTransformation:
    """L-BFGS with zoom linesearch (strong Wolfe conditions).

    This is the standard L-BFGS with zoom linesearch that enforces both:
    - Sufficient decrease (Armijo): f(x + η*d) ≤ f(x) + c1*η*∇f(x)ᵀd
    - Curvature condition: |∇f(x + η*d)ᵀd| ≤ c2*|∇f(x)ᵀd|

    Args:
        learning_rate: Optional global scaling factor.
        memory_size: Number of past updates for Hessian approximation.
        scale_init_precond: Whether to scale initial Hessian approximation.
            WARNING: Set to False for numerically sensitive problems.
        max_linesearch_steps: Maximum iterations for zoom linesearch.
        initial_guess_strategy: "one" (start at η=1) or "keep" (use previous).
        slope_rtol: c1 parameter for Armijo condition (default 1e-4).
        curv_rtol: c2 parameter for curvature condition (default 0.9).
        verbose: Print linesearch debugging info.
        lower: Optional lower bounds for box projection (pytree).
        upper: Optional upper bounds for box projection (pytree).

    Returns:
        An optax GradientTransformation.
    """
    if learning_rate is None:
        base_scaling = transform.scale(-1.0)
    else:
        base_scaling = optax.scale_by_learning_rate(learning_rate)

    linesearch = _linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps,
        initial_guess_strategy=initial_guess_strategy,
        slope_rtol=slope_rtol,
        curv_rtol=curv_rtol,
        verbose=verbose,
    )

    chain_components = [
        transform.scale_by_lbfgs(
            memory_size=memory_size,
            scale_init_precond=scale_init_precond,
        ),
        base_scaling,
        linesearch,
    ]

    # Add projection if bounds provided
    if lower is not None and upper is not None:
        chain_components.append(apply_projection(lower, upper))

    return combine.chain(*chain_components)


def lbfgs_backtrack(
    learning_rate: optax.ScalarOrSchedule | None = None,
    memory_size: int = 10,
    scale_init_precond: bool = False,
    max_backtracking_steps: int = 200,
    slope_rtol: float = 1e-4,
    decrease_factor: float = 0.8,
    increase_factor: float = 1.5,
    max_learning_rate: float = 1.0,
    verbose: bool = False,
    lower: PyTree[Float[Array, " P"]] | None = None,
    upper: PyTree[Float[Array, " P"]] | None = None,
) -> optax.GradientTransformation:
    """L-BFGS with backtracking linesearch (Armijo condition only).

    Simpler than zoom linesearch, only enforces sufficient decrease:
    - Armijo: f(x + η*d) ≤ f(x) + c1*η*∇f(x)ᵀd

    Args:
        learning_rate: Optional global scaling factor.
        memory_size: Number of past updates for Hessian approximation.
        scale_init_precond: Whether to scale initial Hessian approximation.
            WARNING: Set to False for numerically sensitive problems.
        max_backtracking_steps: Maximum backtracking iterations.
        slope_rtol: c1 parameter for Armijo condition (default 1e-4).
        decrease_factor: Multiply stepsize by this when condition fails (default 0.8).
        increase_factor: Initial guess = previous * this factor (default 1.5).
        max_learning_rate: Upper bound on stepsize (default 1.0).
        verbose: Print linesearch debugging info.
        lower: Optional lower bounds for box projection (pytree).
        upper: Optional upper bounds for box projection (pytree).

    Returns:
        An optax GradientTransformation.
    """
    if learning_rate is None:
        base_scaling = transform.scale(-1.0)
    else:
        base_scaling = optax.scale_by_learning_rate(learning_rate)

    linesearch = _linesearch.scale_by_backtracking_linesearch(
        max_backtracking_steps=max_backtracking_steps,
        slope_rtol=slope_rtol,
        decrease_factor=decrease_factor,
        increase_factor=increase_factor,
        max_learning_rate=max_learning_rate,
        verbose=verbose,
    )

    chain_components = [
        transform.scale_by_lbfgs(
            memory_size=memory_size,
            scale_init_precond=scale_init_precond,
        ),
        base_scaling,
        linesearch,
    ]

    # Add projection if bounds provided
    if lower is not None and upper is not None:
        chain_components.append(apply_projection(lower, upper))

    return combine.chain(*chain_components)


# =============================================================================
# BOX PROJECTION TRANSFORMATION
# =============================================================================


def apply_projection(
    lower: PyTree[Float[Array, " P"]] | None = None,
    upper: PyTree[Float[Array, " P"]] | None = None,
) -> optax.GradientTransformation:
    """Wrap box projection into a GradientTransformation.

    After applying this transformation, params + updates will be within [lower, upper].
    The update rule: u_new = clip(p + u, lower, upper) - p

    This can be chained with optimizers like:
        optimizer = optax.chain(
            optax.adam(learning_rate=1e-3),
            apply_projection(lower={'w': 0.0}, upper={'w': 1.0})
        )

    Args:
        lower: Lower bounds (pytree matching params structure)
        upper: Upper bounds (pytree matching params structure)

    Returns:
        GradientTransformation that projects updates to keep params in bounds.
    """

    def init_fn(params: PyTree[Float[Array, " P"]]) -> optax.EmptyState:
        del params
        return optax.EmptyState()

    def update_fn(
        updates: PyTree[Float[Array, " P"]],
        state: optax.EmptyState,
        params: PyTree[Float[Array, " P"]] | None = None,
    ) -> tuple[PyTree[Float[Array, " P"]], optax.EmptyState]:
        if params is None:
            raise ValueError("NO_PARAMS_MSG")

        if lower is None or upper is None:
            return updates, state

        def process_leaf(
            p: Float[Array, " P"],
            u: Float[Array, " P"],
            lo: Float[Array, " P"],
            hi: Float[Array, " P"],
        ) -> Float[Array, " P"]:
            if p is None or u is None:
                return u
            tentative = p + u
            projected = jnp.clip(tentative, lo, hi)
            return projected - p

        new_updates = jax.tree.map(process_leaf, params, updates, lower, upper)
        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


# =============================================================================
# PER-STEP HISTORY LOGGING TRANSFORMATION
# =============================================================================


class LogHistoryState(NamedTuple):
    """State for :func:`log_history`.

    Attributes
    ----------
    count : Int[Array, ""]
        Number of updates logged so far (also the next write index).
    history : Float[Array, "2 N"]
        Pre-allocated table. Row 0 is the update (step) L2 norm, row 1 is the
        objective value, one column per step. Columns at index ``>= count`` are
        still zero.
    """

    count: Int[Array, ""]
    history: Float[Array, "2 N"]


def log_history(length: int) -> optax.GradientTransformationExtraArgs:
    """Record the per-step update norm and objective into a pre-allocated table.

    A pure observer: it returns ``updates`` unchanged and mutates only its own
    state, so chaining it into an optimizer does not perturb the trajectory. It
    must be chained **last** so the logged norm is the fully-transformed step,
    and it relies on ``value=`` being forwarded by ``optimistix.OptaxMinimiser``
    (the objective at the current iterate).

    Args:
        length: Number of steps to pre-allocate (``max_iter``). Writes past this
            are silently dropped by JAX.

    Returns:
        GradientTransformation whose state carries a ``(2, length)`` history table.
    """

    def init_fn(params: PyTree[Float[Array, " P"]]) -> LogHistoryState:
        del params
        return LogHistoryState(count=jnp.array(0), history=jnp.zeros((2, length)))

    def update_fn(
        updates: PyTree[Float[Array, " P"]],
        state: LogHistoryState,
        params: PyTree[Float[Array, " P"]] | None = None,
        value: Float[Array, ""] | None = None,
        **kwargs: Any,
    ) -> tuple[PyTree[Float[Array, " P"]], LogHistoryState]:
        del params, kwargs
        unorm = otu.tree_norm(updates)
        i = state.count
        history = state.history.at[0, i].set(unorm).at[1, i].set(value)
        return updates, LogHistoryState(count=i + 1, history=history)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


# =============================================================================
# SOLVER NAMES AND FACTORY
# =============================================================================

SOLVER_NAMES = Literal[
    # Optax L-BFGS (jax_grid_search compatible)
    "optax_lbfgs",
    "optax_lbfgs",
    "adam",
    "sgd",
    "adabelief",
    "adaw",
    # AdaTopK: "ADABK{N}", matched by prefix in get_solver
    # Optimistix BFGS
    "optimistix_bfgs",
    # Optimistix L-BFGS
    "optimistix_lbfgs",
    # Optimistix NCG (Armijo)
    "optimistix_ncg_pr",
    "optimistix_ncg_hs",
    "optimistix_ncg_fr",
    "optimistix_ncg_dy",
    # Scipy
    "scipy_tnc",
    "scipy_cobyqa",
    # Legacy aliases
    "zoom",
    "backtrack",
]

# Solvers that scale the parameters themselves; minimize() ignores precondition=True for
# them. AdaTopK ("ADABK{N}") is matched by prefix, see is_self_conditioned.
SELFCONDITIONED_SOLVERS = {"scipy_tnc", "scipy_cobyqa"}


def is_self_conditioned(solver_name: str) -> bool:
    """Whether the solver scales the parameters itself (precondition is then ignored)."""
    return solver_name in SELFCONDITIONED_SOLVERS or solver_name.startswith("ADABK")


def get_solver(
    solver_name: SOLVER_NAMES,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    learning_rate: float = 1e-3,
    max_linesearch_steps: int | None = None,
    lower: PyTree[Float[Array, " P"]] | None = None,
    upper: PyTree[Float[Array, " P"]] | None = None,
    verbose_print: bool = False,
    cooldown: int = 20,
    **kwargs: Any,
) -> tuple[Solver, Literal["optimistix", "scipy"]]:
    """
    Create a solver instance from a name string.

    Parameters
    ----------
    solver_name : str
        Solver identifier. See SOLVER_NAMES for available options.
    rtol : float
        Relative tolerance for optimistix solvers.
    atol : float
        Absolute tolerance for optimistix solvers.
    learning_rate : float
        Learning rate for adam solver.
    max_linesearch_steps : int, optional
        Maximum line-search steps per iteration (``optax_lbfgs``, ``sgd``: default 50;
        AdaTopK: default 30, see :class:`cadre.adatopk.AdaTopK`).
    lower : PyTree, optional
        Lower bounds (box projection for the optax solvers, box of AdaTopK).
    upper : PyTree, optional
        Upper bounds (box projection for the optax solvers, box of AdaTopK).
    verbose_print : bool
        If True, AdaTopK prints per-step diagnostics via ``jax.debug.print``
        (JIT-compatible).
    cooldown : int
        Steps to suppress termination after a constraint release (AdaTopK only).
    **kwargs
        Further solver options; for AdaTopK, the fields of :class:`cadre.adatopk.AdaTopK`.

    Returns
    -------
    solver : Solver
        The solver instance: an optimistix minimiser (BestSoFar-wrapped, or AdaTopK), or a
        string for scipy.
    solver_type : str
        One of "optimistix", "scipy".
    """
    # Legacy aliases of optax_lbfgs with a given line search.
    if solver_name in ("zoom", "backtrack"):
        kwargs.setdefault("linesearch", "zoom" if solver_name == "zoom" else "backtracking")
        solver_name = "optax_lbfgs"
    # AdaTopK keeps its own default line-search length; the other solvers use 50.
    ls_steps = 50 if max_linesearch_steps is None else max_linesearch_steps
    # Optax solvers (with optional box projection)
    if solver_name == "optax_lbfgs":
        linesearch_type = kwargs.pop("linesearch", "zoom")
        if linesearch_type == "zoom":
            return optx.BestSoFarMinimiser(
                optx.OptaxMinimiser(
                    lbfgs_zoom(
                        max_linesearch_steps=ls_steps,
                        lower=lower,
                        upper=upper,
                        **kwargs,
                    ),
                    atol=atol,
                    rtol=rtol,
                )
            ), "optimistix"
        elif linesearch_type == "backtracking":
            return optx.BestSoFarMinimiser(
                optx.OptaxMinimiser(
                    lbfgs_backtrack(
                        max_backtracking_steps=ls_steps,
                        lower=lower,
                        upper=upper,
                        **kwargs,
                    ),
                    atol=atol,
                    rtol=rtol,
                )
            ), "optimistix"
        else:
            raise ValueError(
                f"Unknown linesearch type: {linesearch_type}. Use 'backtracking' or 'zoom'."
            )
    elif solver_name == "adam":
        # Chain adam with projection if bounds provided
        # learning_rate from kwargs takes precedence over function parameter
        lr = kwargs.pop("learning_rate", learning_rate)
        adam_opt = optax.adam(learning_rate=lr, **kwargs)
        if lower is not None and upper is not None:
            adam_opt = combine.chain(adam_opt, apply_projection(lower, upper))
        return optx.BestSoFarMinimiser(
            optx.OptaxMinimiser(adam_opt, atol=atol, rtol=rtol)
        ), "optimistix"
    elif solver_name == "sgd":
        # Chain sgd with projection if bounds provided
        # learning_rate from kwargs takes precedence (default 1.0 for linesearch use)
        lr = kwargs.pop("learning_rate", 1.0)
        direction = optax.sgd(learning_rate=lr)
        # Keep your line search
        linesearch = _linesearch.scale_by_backtracking_linesearch(max_backtracking_steps=ls_steps)
        if lower is not None and upper is not None:
            sgd_opt = combine.chain(direction, linesearch, apply_projection(lower, upper))
        else:
            sgd_opt = combine.chain(direction, linesearch)
        return optx.BestSoFarMinimiser(
            optx.OptaxMinimiser(sgd_opt, atol=atol, rtol=rtol)
        ), "optimistix"
    elif solver_name == "adabelief":
        lr = kwargs.pop("learning_rate", learning_rate)
        opt = optax.adabelief(learning_rate=lr)
        if lower is not None and upper is not None:
            opt = combine.chain(opt, apply_projection(lower, upper))
        return optx.BestSoFarMinimiser(optx.OptaxMinimiser(opt, atol=atol, rtol=rtol)), "optimistix"
    elif solver_name == "adaw" or solver_name == "adamw":
        lr = kwargs.pop("learning_rate", learning_rate)
        opt = optax.adamw(learning_rate=lr, **kwargs)
        if lower is not None and upper is not None:
            opt = combine.chain(opt, apply_projection(lower, upper))
        return optx.BestSoFarMinimiser(optx.OptaxMinimiser(opt, atol=atol, rtol=rtol)), "optimistix"
    elif solver_name.startswith("ADABK"):
        # AdaTopK (cadre.adatopk); its own default line-search length unless one is given.
        if solver_name[5:] and not solver_name[5:].isdigit():
            raise ValueError(
                f"Invalid AdaTopK solver name {solver_name!r}: expected 'ADABK' followed by a "
                "non-negative integer N (a fraction N / 10 of the constraints released), e.g. "
                "'ADABK0'."
            )
        if max_linesearch_steps is not None:
            kwargs["max_linesearch_steps"] = max_linesearch_steps
        return AdaTopK(
            rtol=rtol,
            atol=atol,
            lower=lower,
            upper=upper,
            solver_name=solver_name,
            cooldown=cooldown,
            verbose_print=verbose_print,
            **kwargs,
        ), "optimistix"
    # Optimistix BFGS
    elif solver_name == "optimistix_bfgs":
        return optx.BestSoFarMinimiser(optx.BFGS(rtol=rtol, atol=atol, **kwargs)), "optimistix"
    # Optimistix L-BFGS
    elif solver_name == "optimistix_lbfgs":
        return optx.BestSoFarMinimiser(optx.LBFGS(rtol=rtol, atol=atol, **kwargs)), "optimistix"
    # Optimistix NCG (Armijo)
    elif solver_name == "optimistix_ncg_pr":
        return optx.BestSoFarMinimiser(
            optx.NonlinearCG(rtol=rtol, atol=atol, method=optx.polak_ribiere, **kwargs)
        ), "optimistix"
    elif solver_name == "optimistix_ncg_hs":
        return optx.BestSoFarMinimiser(
            optx.NonlinearCG(rtol=rtol, atol=atol, method=optx.hestenes_stiefel, **kwargs)
        ), "optimistix"
    elif solver_name == "optimistix_ncg_fr":
        return optx.BestSoFarMinimiser(
            optx.NonlinearCG(rtol=rtol, atol=atol, method=optx.fletcher_reeves, **kwargs)
        ), "optimistix"
    elif solver_name == "optimistix_ncg_dy":
        return optx.BestSoFarMinimiser(
            optx.NonlinearCG(rtol=rtol, atol=atol, method=optx.dai_yuan, **kwargs)
        ), "optimistix"
    # Scipy (returns string — handled by minimize() via scipy_minimize)
    elif solver_name == "scipy_tnc":
        return "scipy_tnc", "scipy"
    elif solver_name == "scipy_cobyqa":
        return "scipy_cobyqa", "scipy"

    else:
        raise ValueError(f"Unknown solver: {solver_name}. Available: {list(SOLVER_NAMES.__args__)}")
