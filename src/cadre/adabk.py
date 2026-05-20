"""ADABK solver family: AdaBelief + Top-K active set.

ADABK{N} solvers combine an inner AdaBelief direction with the TNC-style
active-set / Top-K constraint-release framework defined in
:mod:`cadre.active_set`. The trailing integer ``N`` controls the fraction of
active constraints released per iteration (``N × 0.1``); ``ADABK0`` releases
exactly one constraint at a time.

References:
    Kabalan et al. (2025), arXiv:2604.08463 — ADABK / AdaTopK active set
    Zhuang et al. (2020), NeurIPS — AdaBelief Optimizer
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import optimistix as optx
from optax._src import linesearch as _linesearch

from .active_set import ActiveSetMinimiser, active_set

# =============================================================================
# Reset function — called by the active-set loop when the pivot vector changes
# =============================================================================


def _reset_adabk_direction(state: optax.OptState) -> optax.OptState:
    """Reset AdaBelief first/second moment estimates after active-set change.

    Zeros ``mu`` (first moment) and ``nu`` (second moment) of the
    ``ScaleByBeliefState`` while leaving ``count`` and other fields intact.
    The next update accumulates fresh moments in the new active subspace.

    References:
        Zhuang et al. (2020), NeurIPS — AdaBelief Optimizer (mu/nu state)
    """
    belief_state = state[0]
    rest = tuple(state[1:])
    new_belief = belief_state._replace(
        mu=jtu.tree_map(jnp.zeros_like, belief_state.mu),
        nu=jtu.tree_map(jnp.zeros_like, belief_state.nu),
    )
    return type(state)((new_belief, *rest))


# =============================================================================
# Solver factory
# =============================================================================


def _parse_adabk_k(solver_name: str) -> float | None:
    """Parse the trailing integer of ``ADABK{N}`` → release fraction ``N × 0.1``."""
    if not solver_name.startswith("ADABK"):
        return None
    if len(solver_name) <= 5:
        return None
    try:
        return int(solver_name[5:]) * 0.1
    except ValueError:
        raise ValueError(
            f"Invalid solver name: {solver_name}. "
            "When using 'ADABK' prefix, it should be followed by an integer."
        )


def make_adabk_solver(
    solver_name: str,
    *,
    rtol: float,
    atol: float,
    max_linesearch_steps: int,
    lower: Any,
    upper: Any,
    verbose_print: bool,
    cooldown: int,
    active_set_extras: dict[str, Any],
    **kwargs: Any,
) -> optx.BestSoFarMinimiser:
    """Build an ADABK active-set minimiser wrapped in ``optx.BestSoFarMinimiser``.

    Recognised kwargs:

    * ``learning_rate`` (float, default 1.0) — AdaBelief learning rate.
    * ``linesearch`` (str, default ``"zoom"``) — ``"zoom"`` or ``"backtracking"``.
    * ``max_constraints_to_release`` (int|float|None) — overrides the
      ``ADABK{N}`` prefix parsing. Float is interpreted as a fraction of total
      params, int as an absolute count.
    """
    lr = kwargs.pop("learning_rate", 1.0)
    linesearch_type = kwargs.pop("linesearch", "zoom")
    max_constraints_to_release = kwargs.pop("max_constraints_to_release", None)
    if max_constraints_to_release is None:
        max_constraints_to_release = _parse_adabk_k(solver_name)

    direction = optax.adabelief(learning_rate=lr)

    if linesearch_type == "backtracking":
        linesearch = _linesearch.scale_by_backtracking_linesearch(
            max_backtracking_steps=max_linesearch_steps
        )
    elif linesearch_type == "zoom":
        linesearch = _linesearch.scale_by_zoom_linesearch(max_linesearch_steps=max_linesearch_steps)
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
                max_constraints_to_release=max_constraints_to_release,
                reset_direction_fn=_reset_adabk_direction,
                verbose_print=verbose_print,
                **active_set_extras,
                **kwargs,
            ),
            atol=atol,
            rtol=rtol,
            cooldown_steps=cooldown,
            verbose_print=verbose_print,
        )
    )
