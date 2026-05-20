"""LBFGSK solver family: L-BFGS + EMA belief factor + Top-K active set.

LBFGSK{N} solvers combine an inner L-BFGS direction (optionally pre-smoothed
by an EMA "belief factor") with the TNC-style active-set / Top-K constraint-
release framework defined in :mod:`cadre.active_set`. The trailing integer
``N`` controls the fraction of active constraints released per iteration
(``N × 0.1``); ``LBFGSK0`` releases exactly one constraint at a time.

LBFGSK is designed as the drop-in successor to ``optax_lbfgs``: it is
competitive on smooth unconstrained problems, strictly better on tight box-
constrained problems where many parameters end up at the walls, and more
robust on noisy gradients when the EMA decay is non-zero.

Key features:

* L-BFGS curvature pairs are automatically reset (count → 0, memory → 0)
  whenever the active set changes (see :func:`_reset_lbfgsk_direction`).
  This avoids polluting the Hessian approximation with pairs from a
  different active subspace.
* The EMA layer (``lbfgs_ema_decay``) smooths the input gradient before it
  reaches the L-BFGS update. ``decay = 0.0`` (default) ⇒ pure L-BFGS. Higher
  decay ⇒ more smoothing, useful for stochastic / low-SNR gradients.
* The active-set framework supplies Langevin noise and Gaussian-prior
  augmentation transparently.

References:
    Moritz et al. (2016), arXiv:1508.02087 — stochastic L-BFGS
    Liu & Nocedal (1989) — limited-memory BFGS
    Schraudolph et al. (2007), AISTATS — online quasi-Newton
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import optimistix as optx
from optax._src import combine, transform
from optax._src import linesearch as _linesearch

from .active_set import ActiveSetMinimiser, active_set

# =============================================================================
# Reset function — called by the active-set loop when the pivot vector changes
# =============================================================================


def _reset_lbfgsk_direction(state: optax.OptState) -> optax.OptState:
    """Reset EMA trace and L-BFGS history after active-set pivot change.

    The L-BFGS curvature subspace is invalidated whenever an active constraint
    is released or hit: gradient/parameter pairs span a different subspace.
    We zero ``count`` and all memory arrays of ``ScaleByLBFGSState`` so the
    next iteration falls back to an identity-Hessian approximation, and zero
    the EMA trace so new curvature pairs accumulate cleanly.

    ``params`` and ``updates`` fields of ``ScaleByLBFGSState`` are deliberately
    NOT zeroed — those snapshots are needed to compute the next curvature
    pair, and zeroing them would inject a huge spurious displacement.

    References:
        Moritz et al. (2016), arXiv:1508.02087 — stochastic L-BFGS
        Liu & Nocedal (1989) — limited-memory BFGS
        Schraudolph et al. (2007), AISTATS — online quasi-Newton
    """
    ema_state = state[0]
    lbfgs_state = state[1]
    rest = tuple(state[2:])

    new_ema = ema_state._replace(ema=jtu.tree_map(jnp.zeros_like, ema_state.ema))
    new_lbfgs = lbfgs_state._replace(
        count=jnp.zeros_like(lbfgs_state.count),
        diff_params_memory=jtu.tree_map(jnp.zeros_like, lbfgs_state.diff_params_memory),
        diff_updates_memory=jtu.tree_map(jnp.zeros_like, lbfgs_state.diff_updates_memory),
        weights_memory=jnp.zeros_like(lbfgs_state.weights_memory),
    )

    return type(state)((new_ema, new_lbfgs, *rest))


# =============================================================================
# Solver factory
# =============================================================================


def _parse_lbfgsk_k(solver_name: str) -> float | None:
    """Parse the trailing integer of ``LBFGSK{N}`` → release fraction ``N × 0.1``."""
    if not solver_name.startswith("LBFGSK"):
        return None
    if len(solver_name) <= 6:
        return None
    try:
        return int(solver_name[6:]) * 0.1
    except ValueError:
        raise ValueError(
            f"Invalid solver name: {solver_name}. "
            "When using 'LBFGSK' prefix, it should be followed by an integer."
        )


def make_lbfgsk_solver(
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
    """Build an LBFGSK active-set minimiser wrapped in ``optx.BestSoFarMinimiser``.

    Recognised kwargs:

    * ``lbfgs_ema_decay`` (float, default 0.0) — EMA smoothing applied to
      the input gradient before the L-BFGS update. ``0.0`` ⇒ pure L-BFGS
      (matches ``optax_lbfgs`` on smooth problems). Set higher (e.g. ``0.9``)
      on low-SNR / stochastic gradients.
    * ``memory_size`` (int, default 10) — L-BFGS history length.
    * ``scale_init_precond`` (bool, default False) — scale the initial
      Hessian approximation. False is safer on numerically sensitive problems.
    * ``linesearch`` (str, default ``"zoom"``) — ``"zoom"`` or ``"backtracking"``.
    * ``max_constraints_to_release`` (int|float|None) — overrides the
      ``LBFGSK{N}`` prefix parsing.
    """
    ema_decay = kwargs.pop("lbfgs_ema_decay", 0.0)
    memory_size = kwargs.pop("memory_size", 10)
    scale_init_precond = kwargs.pop("scale_init_precond", False)
    linesearch_type = kwargs.pop("linesearch", "zoom")
    max_constraints_to_release = kwargs.pop("max_constraints_to_release", None)
    if max_constraints_to_release is None:
        max_constraints_to_release = _parse_lbfgsk_k(solver_name)

    direction = combine.chain(
        optax.ema(decay=ema_decay),
        transform.scale_by_lbfgs(
            memory_size=memory_size,
            scale_init_precond=scale_init_precond,
        ),
        transform.scale(-1.0),
    )

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
                reset_direction_fn=_reset_lbfgsk_direction,
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
