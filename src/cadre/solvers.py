"""Solver factory and dispatch table.

This module wires together the public ``get_solver(name)`` interface. The
active-set families (ADABK and LBFGSK) live in their own modules
(:mod:`cadre.adabk` and :mod:`cadre.lbfgsk`); everything else in this file is
either an off-the-shelf wrapper or a small bound-projection helper.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, Union

import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from jaxtyping import Array, Float, PyTree
from optax._src import combine, transform
from optax._src import linesearch as _linesearch

from .active_set import ActiveSetMinimiser, active_set
from .adabk import _reset_adabk_direction, make_adabk_solver
from .lbfgsk import make_lbfgsk_solver

Solver: TypeAlias = Union[optx.BestSoFarMinimiser, str]


# =============================================================================
# OFF-THE-SHELF L-BFGS factories (kept for use by furax-cs and `optax_lbfgs`)
# =============================================================================


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
    """L-BFGS with zoom linesearch (strong Wolfe conditions)."""
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
    """L-BFGS with backtracking linesearch (Armijo condition only)."""
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

    After applying this transformation, ``params + updates`` will be clipped to
    ``[lower, upper]`` element-wise.
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
            raise ValueError("apply_projection requires `params=` to be passed in update()")

        if lower is None or upper is None:
            return updates, state

        def process_leaf(p, u, lo, hi):
            if p is None or u is None:
                return u
            return jnp.clip(p + u, lo, hi) - p

        return jax.tree.map(process_leaf, params, updates, lower, upper), state

    return optax.GradientTransformation(init_fn, update_fn)


# =============================================================================
# SOLVER NAMES AND FACTORY
# =============================================================================

SOLVER_NAMES = Literal[
    # Optax L-BFGS (jax_grid_search compatible)
    "optax_lbfgs",
    "adam",
    "sgd",
    "adabelief",
    "adaw",
    # Active set families
    "active_set",
    "active_set_sgd",
    "active_set_adabelief",
    "active_set_adaw",
    # Scipy
    "scipy_tnc",
    "scipy_cobyqa",
]

SELFCONDITIONED_SOLVERS = {
    "active_set",
    "active_set_sgd",
    "active_set_adabelief",
    "active_set_adaw",
    "scipy_tnc",
    "scipy_cobyqa",
}


def _is_adabk(name: str) -> bool:
    return name == "active_set_adabelief" or name.startswith("ADABK")


def _is_lbfgsk(name: str) -> bool:
    return name.startswith("LBFGSK")


def get_solver(
    solver_name: SOLVER_NAMES,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    learning_rate: float = 1e-3,
    max_linesearch_steps: int = 50,
    lower: PyTree[Float[Array, " P"]] | None = None,
    upper: PyTree[Float[Array, " P"]] | None = None,
    verbose_print: bool = False,
    cooldown: int = 20,
    **kwargs: Any,
) -> tuple[Solver, Literal["optimistix", "scipy"]]:
    """Create a solver instance from a name string.

    The ``ADABK{N}`` and ``LBFGSK{N}`` families dispatch via prefix to
    :func:`cadre.adabk.make_adabk_solver` and
    :func:`cadre.lbfgsk.make_lbfgsk_solver`. The trailing integer ``N`` is
    parsed as the release fraction ``N × 0.1`` (``ADABK0`` / ``LBFGSK0``
    release one constraint at a time).

    Active-set-only kwargs (``gaussian_prior``, ``noise_temperature``,
    ``noise_decay``, ``noise_key``) are consumed up-front so they never reach
    non-active-set solver factories.

    Self-conditioned solvers (see :data:`SELFCONDITIONED_SOLVERS`) handle
    their own parameter scaling — the unified ``minimize()`` wrapper skips
    its ``precondition`` step for them.
    """
    # Consume active-set-only kwargs up-front.
    gaussian_prior = kwargs.pop("gaussian_prior", None)
    noise_temperature = kwargs.pop("noise_temperature", 0.0)
    noise_decay = kwargs.pop("noise_decay", 1e-3)
    noise_key = kwargs.pop("noise_key", 0)

    active_set_extras = dict(
        gaussian_prior=gaussian_prior,
        noise_temperature=noise_temperature,
        noise_decay=noise_decay,
        noise_key=noise_key,
    )

    # ---- ADABK family ----
    if _is_adabk(solver_name):
        return make_adabk_solver(
            solver_name,
            rtol=rtol,
            atol=atol,
            max_linesearch_steps=max_linesearch_steps,
            lower=lower,
            upper=upper,
            verbose_print=verbose_print,
            cooldown=cooldown,
            active_set_extras=active_set_extras,
            **kwargs,
        ), "optimistix"

    # ---- LBFGSK family ----
    if _is_lbfgsk(solver_name):
        return make_lbfgsk_solver(
            solver_name,
            rtol=rtol,
            atol=atol,
            max_linesearch_steps=max_linesearch_steps,
            lower=lower,
            upper=upper,
            verbose_print=verbose_print,
            cooldown=cooldown,
            active_set_extras=active_set_extras,
            **kwargs,
        ), "optimistix"

    # ---- Off-the-shelf optax_lbfgs ----
    if solver_name == "optax_lbfgs":
        linesearch_type = kwargs.pop("linesearch", "zoom")
        if linesearch_type == "zoom":
            return optx.BestSoFarMinimiser(
                optx.OptaxMinimiser(
                    lbfgs_zoom(
                        max_linesearch_steps=max_linesearch_steps,
                        lower=lower,
                        upper=upper,
                        **kwargs,
                    ),
                    atol=atol,
                    rtol=rtol,
                )
            ), "optimistix"
        if linesearch_type == "backtracking":
            return optx.BestSoFarMinimiser(
                optx.OptaxMinimiser(
                    lbfgs_backtrack(
                        max_backtracking_steps=max_linesearch_steps,
                        lower=lower,
                        upper=upper,
                        **kwargs,
                    ),
                    atol=atol,
                    rtol=rtol,
                )
            ), "optimistix"
        raise ValueError(
            f"Unknown linesearch type: {linesearch_type}. Use 'backtracking' or 'zoom'."
        )

    # ---- First-order optax solvers (adam / sgd / adabelief / adaw) ----
    if solver_name == "adam":
        lr = kwargs.pop("learning_rate", learning_rate)
        adam_opt = optax.adam(learning_rate=lr, **kwargs)
        if lower is not None and upper is not None:
            adam_opt = combine.chain(adam_opt, apply_projection(lower, upper))
        return optx.BestSoFarMinimiser(
            optx.OptaxMinimiser(adam_opt, atol=atol, rtol=rtol)
        ), "optimistix"
    if solver_name == "sgd":
        lr = kwargs.pop("learning_rate", 1.0)
        direction = optax.sgd(learning_rate=lr)
        linesearch = _linesearch.scale_by_backtracking_linesearch(
            max_backtracking_steps=max_linesearch_steps
        )
        if lower is not None and upper is not None:
            sgd_opt = combine.chain(direction, linesearch, apply_projection(lower, upper))
        else:
            sgd_opt = combine.chain(direction, linesearch)
        return optx.BestSoFarMinimiser(
            optx.OptaxMinimiser(sgd_opt, atol=atol, rtol=rtol)
        ), "optimistix"
    if solver_name == "adabelief":
        lr = kwargs.pop("learning_rate", learning_rate)
        opt = optax.adabelief(learning_rate=lr)
        if lower is not None and upper is not None:
            opt = combine.chain(opt, apply_projection(lower, upper))
        return optx.BestSoFarMinimiser(optx.OptaxMinimiser(opt, atol=atol, rtol=rtol)), "optimistix"
    if solver_name in ("adaw", "adamw"):
        lr = kwargs.pop("learning_rate", learning_rate)
        opt = optax.adamw(learning_rate=lr, **kwargs)
        if lower is not None and upper is not None:
            opt = combine.chain(opt, apply_projection(lower, upper))
        return optx.BestSoFarMinimiser(optx.OptaxMinimiser(opt, atol=atol, rtol=rtol)), "optimistix"

    # ---- Custom-direction active-set variants (Adam / SGD / AdamW) ----
    if solver_name in ("active_set", "active_set_sgd", "active_set_adaw"):
        lr = kwargs.pop("learning_rate", 1.0)
        linesearch_type = kwargs.pop("linesearch", "backtracking")

        if solver_name == "active_set":
            direction = optax.adam(learning_rate=lr)
            reset_fn = _reset_adabk_direction  # Adam state mu/nu shape matches AdaBelief.
        elif solver_name == "active_set_sgd":
            direction = optax.sgd(learning_rate=lr)
            reset_fn = None
        else:  # active_set_adaw
            direction = optax.adamw(learning_rate=lr)
            reset_fn = None

        if linesearch_type == "backtracking":
            linesearch = _linesearch.scale_by_backtracking_linesearch(
                max_backtracking_steps=max_linesearch_steps
            )
        elif linesearch_type == "zoom":
            linesearch = _linesearch.scale_by_zoom_linesearch(
                max_linesearch_steps=max_linesearch_steps
            )
        else:
            raise ValueError(
                f"Unknown linesearch type: {linesearch_type}. Use 'backtracking' or 'zoom'."
            )

        return optx.BestSoFarMinimiser(
            ActiveSetMinimiser(
                active_set(
                    direction,
                    linesearch,
                    lower=lower,
                    upper=upper,
                    reset_direction_fn=reset_fn,
                    verbose_print=verbose_print,
                    **active_set_extras,
                    **kwargs,
                ),
                atol=atol,
                rtol=rtol,
                cooldown_steps=cooldown,
                verbose_print=verbose_print,
            )
        ), "optimistix"

    # ---- Scipy backends ----
    if solver_name == "scipy_tnc":
        return "scipy_tnc", "scipy"
    if solver_name == "scipy_cobyqa":
        return "scipy_cobyqa", "scipy"
    if solver_name == "scipy_lbfgsb":
        return "scipy_lbfgsb", "scipy"
    if solver_name == "scipy_trust-constr":
        return "scipy_trust-constr", "scipy"

    raise ValueError(f"Unknown solver: {solver_name}. Available: {list(SOLVER_NAMES.__args__)}")
