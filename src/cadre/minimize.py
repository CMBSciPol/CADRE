from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jaxtyping import (
    Array,
    Float,
    PyTree,  # pyright: ignore
    Scalar,
)

from ._compat import requires_scipy
from ._logging import warning
from .constraints import BoxConstraint, Constraint, GaussianConstraint, validate_constraint
from .solvers import SELFCONDITIONED_SOLVERS, SOLVER_NAMES, get_solver
from .utils import condition

try:
    import jaxopt  # noqa: F401
except ImportError:
    pass

# =============================================================================
# SCIPY MINIMIZE WITH VMAP SUPPORT
# =============================================================================


class ScipyMinimizeState(eqx.Module):
    """State returned by scipy minimize via pure_callback.

    This equinox module holds the optimization result in a JAX-compatible format
    that can be used with vmap/lax.map.

    Attributes
    ----------
    params : PyTree
        Optimized parameters.
    fun_val : Scalar
        Final objective function value (scalar).
    success : Scalar
        Whether optimization converged successfully (bool scalar).
    iter_num : Scalar
        Number of iterations performed (int32 scalar).
    """

    params: PyTree[Float[Array, " P"]]
    fun_val: Scalar
    success: Scalar
    iter_num: Scalar


@requires_scipy
def scipy_minimize(
    fn: Callable[..., Scalar],
    init_params: PyTree[Float[Array, " P"]],
    lower_bound: PyTree[Float[Array, " P"]] | None = None,
    upper_bound: PyTree[Float[Array, " P"]] | None = None,
    method: str = "tnc",
    maxiter: int = 1000,
    **fn_kwargs: Any,
) -> ScipyMinimizeState:
    """Scipy minimize wrapper that supports vmap via jax.pure_callback.

    This function wraps scipy optimization in a way that is compatible with
    JAX transformations like vmap and lax.map. It uses jax.pure_callback to
    call the host-side scipy solver.

    Parameters
    ----------
    fn : Callable
        Objective function to minimize. Should accept (params, **fn_kwargs).
    init_params : PyTree
        Initial parameter values.
    lower_bound : PyTree, optional
        Lower bounds for parameters. Same shape as init_params.
    upper_bound : PyTree, optional
        Upper bounds for parameters. Same shape as init_params.
    method : str
        Scipy optimization method (default "tnc").
    maxiter : int
        Maximum number of iterations.
    **fn_kwargs
        Additional arguments passed to fn.

    Returns
    -------
    ScipyMinimizeState
        Optimization result containing params, fun_val, success, and iter_num.

    Raises
    ------
    ImportError
        If ``cadre[scipy]`` optional dependencies are not installed.
    """
    from jaxopt import ScipyBoundedMinimize

    def host_solver_callback(x_init, lower, upper, fn_kwargs):
        """Host-side scipy solver callback."""
        # Handle bounds
        if lower is None and upper is None:
            bounds = None
        else:
            bounds = (lower, upper)

        # Define wrapped objective
        def scipy_fn(params, fn_kwargs):
            return fn(params, **fn_kwargs)

        # Scipy method handling
        solver_options = {"disp": False}
        if method == "cobyqa":
            try:
                import cobyqa  # noqa: F401
            except ImportError:
                raise ImportError(
                    "cobyqa not installed. Install with: pip install jax-cadre[scipy]"
                )

        solver = ScipyBoundedMinimize(
            fun=scipy_fn,
            method=method,
            jit=False,
            maxiter=maxiter,
            options=solver_options,
        )

        res = solver.run(x_init, bounds=bounds, fn_kwargs=fn_kwargs)

        # Return numpy arrays for pure_callback
        return {
            "params": jax.tree.map(lambda x: np.array(x), res.params),
            "fun_val": np.array(res.state.fun_val, dtype=np.float32),
            "success": np.array(res.state.success, dtype=bool),
            "iter_num": np.array(res.state.iter_num, dtype=np.int32),
        }

    # Define result shape for pure_callback
    result_shape = {
        "params": jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), init_params),
        "fun_val": jax.ShapeDtypeStruct((), jnp.float32),
        "success": jax.ShapeDtypeStruct((), jnp.bool_),
        "iter_num": jax.ShapeDtypeStruct((), jnp.int32),
    }

    result_dict = jax.pure_callback(
        host_solver_callback,
        result_shape,
        init_params,
        lower_bound,
        upper_bound,
        fn_kwargs,
        vmap_method="sequential",
    )

    return ScipyMinimizeState(
        params=result_dict["params"],
        fun_val=result_dict["fun_val"],
        success=result_dict["success"],
        iter_num=result_dict["iter_num"],
    )


# =============================================================================
# UNIFIED STATE
# =============================================================================


class UnifiedState(eqx.Module):
    """Unified optimization state.

    Attributes
    ----------
    best_loss : Scalar
        Best objective function value found.
    best_y : PyTree
        Best parameters found (in original space).
    iter_num : Scalar
        Number of iterations performed.
    solver_state : Any
        Internal solver state (Optimistix state or ScipyMinimizeState).
    """

    best_loss: Scalar
    best_y: PyTree[Float[Array, " P"]]
    iter_num: Scalar
    solver_state: Any


# =============================================================================
# UNIFIED OPTIMIZATION INTERFACE
# =============================================================================


def minimize(
    fn: Callable[..., Scalar],
    init_params: PyTree[Float[Array, " P"]],
    solver_name: SOLVER_NAMES = "optax_lbfgs",
    max_iter: int = 1000,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    constraints: Constraint | None = None,
    precondition: bool = False,
    options: dict[str, Any] | None = None,
    refresh_steps: int = 10,
    *,
    lower_bound: PyTree[Float[Array, " P"]] | None = None,
    upper_bound: PyTree[Float[Array, " P"]] | None = None,
    **fn_kwargs: Any,
) -> tuple[PyTree[Float[Array, " P"]], UnifiedState]:
    """
    Unified optimization interface.

    Supports optax solvers, optimistix solvers (via ``optimistix.minimise``),
    and scipy solvers (via ``jaxopt.ScipyBoundedMinimize`` inside a
    ``jax.pure_callback``; requires ``jax-cadre[scipy]``).

    Parameters
    ----------
    fn : Callable
        Objective function to minimize. Should accept ``(params, **fn_kwargs)``.
    init_params : PyTree
        Initial parameter values.
    solver_name : str
        Solver identifier. See ``SOLVER_NAMES`` for the registered names. In
        addition the prefix-dispatched families ``ADABK{N}`` and ``LBFGSK{N}``
        are accepted (e.g. ``"ADABK0"``, ``"LBFGSK5"``).
    max_iter : int
        Maximum iterations.
    rtol, atol : float
        Relative / absolute tolerance for convergence.
    constraints : Constraint, optional
        The **only** sanctioned way to pass bounds or priors.
        ``BoxConstraint(lower, upper)`` for hard box bounds, or
        ``GaussianConstraint(loc, scale)`` for a soft ridge prior.
        ``GaussianConstraint`` wraps ``fn`` with the prior log-likelihood
        ``½·Σ((x−loc)/scale)²`` so that the gradient, the value, and
        ``BestSoFarMinimiser`` all see the joint MAP objective.
    lower_bound, upper_bound : PyTree, optional, **DEPRECATED**
        Legacy box-constraint kwargs (keyword-only). If supplied, they are
        transparently converted to ``constraints=BoxConstraint(...)`` and a
        :class:`DeprecationWarning` is emitted. Will be removed in a future
        release — pass ``constraints=BoxConstraint(lower=..., upper=...)``
        instead.
    precondition : bool
        Apply min-max parameter scaling to ``[0, 1]`` and rescale ``fn`` by
        ``1/‖∇f(x₀)‖``. Skipped for self-conditioned solvers (see
        :data:`SELFCONDITIONED_SOLVERS`).
    options : dict, optional
        Per-solver knobs. Unknown keys are forwarded to the solver factory.

        Common to **all active-set solvers** (``ADABK{N}``, ``LBFGSK{N}``,
        ``active_set``, ``active_set_sgd``, ``active_set_adabelief``,
        ``active_set_adaw``):

        * ``cooldown`` (int, default 20) — number of steps to suppress
          termination after a constraint release; absorbs transient spikes.
        * ``verbose_print`` (bool, default False) — per-step diagnostics via
          ``jax.debug.print`` (JIT-compatible).
        * ``max_linesearch_steps`` (int, default 50) — line-search budget per
          iteration.
        * ``linesearch`` (str) — ``"zoom"`` (strong Wolfe) or
          ``"backtracking"`` (Armijo only).
        * ``noise_temperature`` (float, default 0.0) — Langevin noise
          amplitude. Effective scale ``T·exp(-decay·step) / (‖pk‖ + ε)``:
          large when stuck, small when the step is informative. ``0.0``
          disables noise (deterministic).
        * ``noise_decay`` (float, default 1e-3) — exponential decay rate of
          the noise temperature with iteration count.
        * ``noise_key`` (int, default 0) — PRNG seed for Langevin noise.
        * ``max_constraints_to_release`` (int|float|None) — overrides the
          ``{N}`` prefix; ``int`` is an absolute count, ``float`` a fraction.

        **LBFGSK{N}-only** keys:

        * ``lbfgs_ema_decay`` (float, default 0.0) — EMA "belief factor"
          applied to gradients before L-BFGS curvature update. ``0.0`` ⇒
          pure L-BFGS. Set to ``0.9`` (or similar) on noisy / stochastic
          gradients; falls back to ``optax_lbfgs`` behaviour at 0.
        * ``memory_size`` (int, default 10) — L-BFGS history length.
        * ``scale_init_precond`` (bool, default False) — scale the initial
          Hessian approximation. ``False`` is safer on ill-conditioned
          problems.

        **ADABK{N}-only** keys:

        * ``learning_rate`` (float, default 1.0) — AdaBelief learning rate.

        **``optax_lbfgs`` keys** (off-the-shelf L-BFGS, box projection):

        * ``linesearch`` — same options as above.
        * Any ``lbfgs_zoom`` / ``lbfgs_backtrack`` factory kwarg
          (``memory_size``, ``scale_init_precond``, ``initial_guess_strategy``,
          ``slope_rtol``, ``curv_rtol``, ``verbose``, etc.).

        **First-order optax keys** (``adam`` / ``sgd`` / ``adabelief`` /
        ``adaw``):

        * ``learning_rate`` (float) — base learning rate. Box constraints are
          enforced by chaining ``apply_projection``.

        **Optimistix keys** (``optimistix_bfgs`` / ``optimistix_lbfgs`` /
        ``optimistix_ncg_*``): forwarded to the underlying ``optx`` solver.

        **Scipy keys** (``scipy_tnc`` / ``scipy_cobyqa``): tolerance keys are
        mapped per-method by ``minimize()``:

        * ``scipy_tnc``: ``ftol = atol``, ``gtol = rtol``, ``xtol = atol``.
        * ``scipy_cobyqa``: ``final_tr_radius = atol``.

    refresh_steps : int
        Progress-meter refresh period for optimistix solvers (best-effort).
    **fn_kwargs
        Forwarded to ``fn``.

    Returns
    -------
    final_params : PyTree
        Optimized parameters in original (un-preconditioned) space.
    final_state : UnifiedState
        Wraps ``best_loss``, ``best_y``, ``iter_num``, and the raw solver state.
    """
    solver_name = cast(SOLVER_NAMES, solver_name)

    # --- Deprecated ``lower_bound`` / ``upper_bound`` kwargs ---
    # Transparently convert to ``constraints=BoxConstraint(...)``.
    if lower_bound is not None or upper_bound is not None:
        import warnings as _warnings

        if constraints is not None:
            raise ValueError(
                "minimize(): pass `constraints=` only — "
                "`lower_bound` / `upper_bound` are deprecated and cannot be combined "
                "with `constraints`."
            )
        _warnings.warn(
            "`lower_bound` / `upper_bound` are deprecated in cadre.minimize(); "
            "pass `constraints=BoxConstraint(lower=..., upper=...)` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Fill in unspecified side with ±inf of the right pytree shape.
        if lower_bound is None:
            lower_bound = jax.tree.map(lambda x: jnp.full_like(x, -jnp.inf), init_params)
        if upper_bound is None:
            upper_bound = jax.tree.map(lambda x: jnp.full_like(x, jnp.inf), init_params)
        constraints = BoxConstraint(lower=lower_bound, upper=upper_bound)

    # --- Constraint dispatch ---
    # ``constraints`` is now the SOLE supported way to specify bounds / priors.
    # Internal variables ``_lower_internal`` / ``_upper_internal`` are populated
    # from a BoxConstraint and passed to the solver factories.
    _lower_internal: PyTree[Float[Array, " P"]] | None = None
    _upper_internal: PyTree[Float[Array, " P"]] | None = None
    gaussian_prior: GaussianConstraint | None = None
    if constraints is not None:
        validate_constraint(constraints)
        if isinstance(constraints, BoxConstraint):
            _lower_internal = constraints.lower
            _upper_internal = constraints.upper
        elif isinstance(constraints, GaussianConstraint):
            gaussian_prior = constraints
        else:
            raise TypeError(f"Unsupported constraint type: {type(constraints).__name__}")

    # If a Gaussian prior is requested, wrap ``fn`` to include the prior log-
    # likelihood so the optimizer and ``BestSoFarMinimiser`` track the joint
    # MAP objective rather than just the data term.
    if gaussian_prior is not None:
        import jax.tree_util as _jtu

        _gp = gaussian_prior
        _orig_fn = fn

        def fn(x, _orig=_orig_fn, _gp=_gp, **kw):  # type: ignore[no-redef]
            data = _orig(x, **kw)
            terms = _jtu.tree_map(
                lambda v, loc, sc: jnp.sum(((v - loc) / sc) ** 2),
                x,
                _gp.loc,
                _gp.scale,
            )
            prior = 0.5 * sum(_jtu.tree_leaves(terms))
            return data + prior

    if solver_name in SELFCONDITIONED_SOLVERS and precondition:
        warning(f"Solver '{solver_name}' is self-conditioned; ignoring preconditioning request.")
        precondition = False

    if precondition:
        fn, to_opt, from_opt = condition(
            fn,
            lower=_lower_internal,
            upper=_upper_internal,
            scale_function=precondition,
            init_params=init_params,
            **fn_kwargs,
        )
        init_params = to_opt(init_params)
        _lower_internal = to_opt(_lower_internal) if _lower_internal is not None else None
        _upper_internal = to_opt(_upper_internal) if _upper_internal is not None else None
    else:
        from_opt = lambda x: x

    _opts = options or {}
    cooldown = _opts.get("cooldown", 20)
    solver_kwargs = {k: v for k, v in _opts.items() if k != "cooldown"}
    if gaussian_prior is not None:
        solver_kwargs.setdefault("gaussian_prior", gaussian_prior)
    solver, solver_type = get_solver(
        solver_name,
        rtol=rtol,
        atol=atol,
        lower=_lower_internal,
        upper=_upper_internal,
        cooldown=cooldown,
        **solver_kwargs,
    )

    if solver_type == "optimistix":
        # Optimistix uses (y, args) signature, wrap fn
        def optx_fn(y, fn_kwargs):
            return fn(y, **fn_kwargs)

        # Does optax have TqdmProgressMeter? defined?
        if not hasattr(optx, "TqdmProgressMeter"):
            kwargs = {}
            warning("optx.TqdmProgressMeter not found. Progress meter disabled.")
        else:
            kwargs = {"progress_meter": optx.TqdmProgressMeter(refresh_steps=refresh_steps)}

        sol = optx.minimise(
            optx_fn,
            solver,
            init_params,
            max_steps=max_iter,
            throw=False,
            args=fn_kwargs,
            **kwargs,
        )

        unified_state = UnifiedState(
            best_loss=sol.state.best_loss,
            best_y=from_opt(sol.state.best_y),
            iter_num=sol.stats["num_steps"],
            solver_state=sol.state,
        )
        return from_opt(sol.value), unified_state

    elif solver_type == "scipy":
        # Scipy via vmap-compatible scipy_minimize
        method = solver_name.split("_")[1]
        options = _opts
        if method == "tnc":
            options["ftol"] = atol
            options["gtol"] = rtol
            options["xtol"] = atol
        elif method == "l-bfgs-b":
            options["ftol"] = atol
            options["gtol"] = rtol
        elif method == "cobyqa":  # COBYQA
            options["final_tr_radius"] = atol
        elif method == "trust-constr":
            options["gtol"] = rtol
            options["xtol"] = atol
        state = scipy_minimize(
            fn=fn,
            init_params=init_params,
            lower_bound=_lower_internal,
            upper_bound=_upper_internal,
            method=method,
            maxiter=max_iter,
            **fn_kwargs,
        )

        unified_state = UnifiedState(
            best_loss=state.fun_val,
            best_y=from_opt(state.params),
            iter_num=state.iter_num,
            solver_state=state,
        )

        return from_opt(state.params), unified_state

    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
