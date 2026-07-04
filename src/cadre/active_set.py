from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import optax.tree_utils as otu
import optimistix as optx
from jax.flatten_util import ravel_pytree
from jaxtyping import (
    Array,
    Bool,
    Float,
    Int,
    PyTree,  # pyright: ignore
    Scalar,
)
from optax._src import linesearch as _linesearch

from ._logging import info

# --- Helper Logic ---


def _compute_initial_pivot(
    y: PyTree[Float[Array, " P"]],
    lower: PyTree[Float[Array, " P"]],
    upper: PyTree[Float[Array, " P"]],
    scale: PyTree[Float[Array, " P"]],
    offset: PyTree[Float[Array, " P"]],
) -> PyTree[Int[Array, " P"]]:
    """Compute initial pivot based on position relative to bounds."""

    def _leaf_pivot(
        y_leaf: Float[Array, " P"],
        lo: Float[Array, " P"],
        up: Float[Array, " P"],
        sc: Float[Array, " P"],
        off: Float[Array, " P"],
    ) -> Int[Array, " P"]:
        EPS = 1e-8  # Slightly relaxed tolerance for float32/64 stability
        tol_lower = EPS * 10.0 * (jnp.abs(lo) + 1.0)
        tol_upper = EPS * 10.0 * (jnp.abs(up) + 1.0)

        is_constant = (sc == 0.0) | (lo == up)

        # Calculate physical Y to check bounds
        y_phys = y_leaf * sc + off

        # Check bounds only if they are finite
        is_finite_lower = lo > -1e20
        is_finite_upper = up < 1e20

        at_lower = is_finite_lower & (y_phys - lo <= tol_lower) & ~is_constant
        at_upper = is_finite_upper & (up - y_phys <= tol_upper) & ~is_constant

        p = jnp.zeros_like(y_leaf, dtype=jnp.int32)
        p = jnp.where(at_lower, -1, p)
        p = jnp.where(at_upper, 1, p)
        p = jnp.where(is_constant, 2, p)
        return p

    return jtu.tree_map(_leaf_pivot, y, lower, upper, scale, offset)


def _compute_step_max(
    step_limit: Scalar,
    y_int: PyTree[Float[Array, " P"]],
    direction: PyTree[Float[Array, " P"]],
    pivot: PyTree[Int[Array, " P"]],
    lower: PyTree[Float[Array, " P"]],
    upper: PyTree[Float[Array, " P"]],
    scale: PyTree[Float[Array, " P"]],
    offset: PyTree[Float[Array, " P"]],
) -> Scalar:
    """Compute max step size alpha such that y + alpha * d stays in bounds."""

    def _leaf_step(
        y_leaf: Float[Array, " P"],
        d_leaf: Float[Array, " P"],
        p_leaf: Int[Array, " P"],
        lo: Float[Array, " P"],
        up: Float[Array, " P"],
        sc: Float[Array, " P"],
        off: Float[Array, " P"],
    ) -> Float[Array, " P"]:
        # Internal bounds
        lo_int = (lo - off) / jnp.where(sc == 0, 1.0, sc)
        up_int = (up - off) / jnp.where(sc == 0, 1.0, sc)

        t_lower = jnp.where(d_leaf < -1e-12, (lo_int - y_leaf) / d_leaf, jnp.inf)
        t_upper = jnp.where(d_leaf > 1e-12, (up_int - y_leaf) / d_leaf, jnp.inf)

        return jnp.minimum(t_lower, t_upper)

    max_steps = jtu.tree_map(_leaf_step, y_int, direction, pivot, lower, upper, scale, offset)

    flat_steps = jtu.tree_leaves(max_steps)
    if not flat_steps:
        return step_limit

    dist_to_bound = jnp.min(jnp.stack([jnp.min(s) for s in flat_steps]))

    return jnp.minimum(step_limit, dist_to_bound)


def _update_pivot_at_boundary(
    y_int: PyTree[Float[Array, " P"]],
    direction: PyTree[Float[Array, " P"]],
    pivot: PyTree[Int[Array, " P"]],
    lower: PyTree[Float[Array, " P"]],
    upper: PyTree[Float[Array, " P"]],
    scale: PyTree[Float[Array, " P"]],
    offset: PyTree[Float[Array, " P"]],
    step_size: Scalar,
) -> PyTree[Int[Array, " P"]]:
    """Update pivot if we landed exactly on a boundary."""

    def _leaf_add(
        y_leaf: Float[Array, " P"],
        d_leaf: Float[Array, " P"],
        p_leaf: Int[Array, " P"],
        lo: Float[Array, " P"],
        up: Float[Array, " P"],
        sc: Float[Array, " P"],
        off: Float[Array, " P"],
    ) -> Int[Array, " P"]:
        y_next = y_leaf + step_size * d_leaf
        y_next_phys = y_next * sc + off

        EPS = 1e-8
        tol_lower = EPS * 10.0 * (jnp.abs(lo) + 1.0)
        tol_upper = EPS * 10.0 * (jnp.abs(up) + 1.0)

        is_free = p_leaf == 0
        hits_lower = is_free & (d_leaf < 0) & (y_next_phys - lo <= tol_lower)
        hits_upper = is_free & (d_leaf > 0) & (up - y_next_phys <= tol_upper)

        new_p = p_leaf
        new_p = jnp.where(hits_lower, -1, new_p)
        new_p = jnp.where(hits_upper, 1, new_p)
        return new_p

    return jtu.tree_map(
        lambda y, d, p, l, u, s, o: _leaf_add(y, d, p, l, u, s, o),
        y_int,
        direction,
        pivot,
        lower,
        upper,
        scale,
        offset,
    )


def _tree_top_k(tree: PyTree, k: int) -> PyTree:
    """Find boolean mask of the top K largest values across an entire PyTree."""
    flat_data, unravel_fn = ravel_pytree(tree)
    n_params = flat_data.shape[0]
    _, top_indices = jax.lax.top_k(flat_data, k)
    flat_mask = jnp.zeros(n_params, dtype=bool)
    flat_mask = flat_mask.at[top_indices].set(True)
    return unravel_fn(flat_mask)


def _release_constraints(
    pivot: PyTree[Int[Array, " P"]],
    gradients_int: PyTree[Float[Array, " P"]],
    max_release_k: int,
) -> tuple[PyTree[Int[Array, " P"]], Bool[Array, ""]]:
    """Release constraints if the negative gradient points into the feasible region.

    Score = pivot * gradient. Releases the Top-K active constraints with the
    strongest positive score (strongest desire to release).

    References:
        Kabalan et al. (2025), arXiv:2604.08463 — AdaTopK active-set release rule
    """

    def _compute_score(p: Int[Array, " P"], g: Float[Array, " P"]) -> Float[Array, " P"]:
        score = p * g
        return jnp.where(jnp.abs(p) == 1, score, -jnp.inf)

    scores = jtu.tree_map(_compute_score, pivot, gradients_int)

    flat_scores, _ = ravel_pytree(scores)
    constraints_released = jnp.any(flat_scores > 0)

    top_k_mask = _tree_top_k(scores, max_release_k)

    def _apply_release(
        p: Int[Array, " P"], is_top_k: Bool[Array, " P"], s: Float[Array, " P"]
    ) -> Int[Array, " P"]:
        should_release = is_top_k & (s > 0)
        return jnp.where(should_release, 0, p)

    return jtu.tree_map(_apply_release, pivot, top_k_mask, scores), constraints_released


# --- Active Set Component ---
def _rescale_adam_state(state: optax.OptState, scale_factor: Scalar) -> optax.OptState:
    """Recursively search for Adam/AdaBelief states and rescale moments.

    Robustly handles optax.chain (tuples/lists) and leaf states (NamedTuples).
    """
    # 1. Identify Adam/AdaBelief State (Target)
    if hasattr(state, "mu") and hasattr(state, "nu"):
        return state._replace(
            mu=otu.tree_scale(scale_factor, state.mu), nu=otu.tree_scale(scale_factor**2, state.nu)
        )

    # 2. Recurse into Containers (optax.chain uses plain tuples/lists)
    # CRITICAL FIX: We must exclude NamedTuples (like EmptyState) from this check.
    elif isinstance(state, tuple | list) and not hasattr(state, "_fields"):
        return type(state)(_rescale_adam_state(s, scale_factor) for s in state)

    # 3. Leave everything else alone (EmptyState, ScheduleState, etc.)
    return state


class ActiveSetState(NamedTuple):
    count: Scalar
    pivot: PyTree[Int[Array, " P"]]
    xscale: PyTree[Float[Array, " P"]]
    offset: PyTree[Float[Array, " P"]]
    lower: PyTree[Float[Array, " P"]]
    upper: PyTree[Float[Array, " P"]]
    fscale: Scalar
    stepmx: Scalar
    max_release_k: Scalar
    direction_state: optax.OptState
    linesearch_state: optax.OptState
    constraints_released: Bool[Array, ""]
    last_release_step: Int[Array, ""]
    # TNC-inspired termination fields
    best_f: Scalar  # best function value seen so far
    f_val: Scalar  # current function value
    prev_f: Scalar  # previous function value
    # KKT termination: physical-space projected gradient norm
    gnorm_proj: Scalar


def active_set(
    direction_solver: optax.GradientTransformation,
    linesearch_solver: optax.GradientTransformation,
    lower: PyTree[Float[Array, " P"]] | None = None,
    upper: PyTree[Float[Array, " P"]] | None = None,
    rescale_threshold: float = 1.3,
    stepmx_init: float = 10.0,
    max_constraints_to_release: Union[int, float] | None = None,
    verbose_print: bool = False,
) -> optax.GradientTransformation:
    """Active-set descent transformation with TNC-style scaling.

    Maps physical parameters x ∈ [lower, upper] to a normalized internal space
    y = (x − offset) / xscale, tracks per-parameter pivot flags
    (0=free, ±1=at bound, 2=constant), and at each step releases the Top-K
    active constraints whose negative gradient points into the feasible region.

    References:
        Kabalan et al. (2025), arXiv:2604.08463 — ADABK / AdaTopK active set
    """

    def init_fn(params: PyTree[Float[Array, " P"]]) -> ActiveSetState:
        lo = lower if lower is not None else otu.tree_full_like(params, -jnp.inf)
        up = upper if upper is not None else otu.tree_full_like(params, jnp.inf)

        leaves = jtu.tree_leaves(params)
        total_params = sum(leaf.size for leaf in leaves)

        if max_constraints_to_release is None:
            k_val = max(1, total_params // 10)
        elif isinstance(max_constraints_to_release, float):
            k_val = min(total_params, max(1, int(total_params * max_constraints_to_release)))
        else:
            k_val = min(max_constraints_to_release, total_params)
            k_val = max(1, k_val)

        info(f"key active_set: max_constraints_to_release={k_val} / {total_params} params")

        # Init Scale & Offset logic from TNC
        def _init_scale(
            p: Float[Array, " P"], l: Float[Array, " P"], u: Float[Array, " P"]
        ) -> Float[Array, " P"]:
            is_bounded = (l > -1e20) & (u < 1e20)
            s_b = u - l
            s_u = 1.0 + jnp.abs(p)
            return jnp.where(is_bounded, s_b, s_u)

        def _init_offset(
            p: Float[Array, " P"], l: Float[Array, " P"], u: Float[Array, " P"]
        ) -> Float[Array, " P"]:
            is_bounded = (l > -1e20) & (u < 1e20)
            o_b = (l + u) * 0.5
            o_u = p
            return jnp.where(is_bounded, o_b, o_u)

        xscale = jtu.tree_map(_init_scale, params, lo, up)
        offset = jtu.tree_map(_init_offset, params, lo, up)

        y_int = otu.tree_div(otu.tree_sub(params, offset), xscale)
        pivot = _compute_initial_pivot(y_int, lo, up, xscale, offset)

        return ActiveSetState(
            count=jnp.array(0, dtype=jnp.int32),
            pivot=pivot,
            xscale=xscale,
            offset=offset,
            lower=lo,
            upper=up,
            fscale=jnp.array(1.0),
            stepmx=jnp.array(stepmx_init),
            max_release_k=k_val,
            direction_state=direction_solver.init(params),
            linesearch_state=linesearch_solver.init(params),
            constraints_released=jnp.array(False),
            last_release_step=jnp.array(-1, dtype=jnp.int32),
            best_f=jnp.array(jnp.inf),
            f_val=jnp.array(jnp.inf),
            prev_f=jnp.array(jnp.inf),
            gnorm_proj=jnp.array(jnp.inf),
        )

    def update_fn(
        grads: PyTree[Float[Array, " P"]],
        state: ActiveSetState,
        params: PyTree[Float[Array, " P"]] | None = None,
        value: Scalar | None = None,
        grad: PyTree[Float[Array, " P"]] | None = None,
        value_fn: Callable[[PyTree[Float[Array, " P"]]], Scalar] | None = None,
        **kwargs: Any,
    ) -> tuple[PyTree[Float[Array, " P"]], ActiveSetState]:
        if params is None or value_fn is None:
            raise ValueError("active_set requires 'params' and 'value_fn' arguments.")
        if value is None:
            value = value_fn(params)

        # --- 1. Internal Representation ---
        y_int = otu.tree_div(otu.tree_sub(params, state.offset), state.xscale)

        # Scale Gradients to Internal Space: g_int = g_phys * xscale * fscale
        grads_int = otu.tree_scale(state.fscale, otu.tree_mul(grads, state.xscale))

        # --- 2. Release Active Constraints ---
        pivot, constraints_released = _release_constraints(
            state.pivot, grads_int, state.max_release_k
        )

        # --- 3. Project Gradients (Input Masking) ---
        grads_int_proj = jax.tree.map(lambda p, pk: jnp.where(p == 0, pk, 0.0), pivot, grads_int)

        # Internal-space projected gradient norm — used below for dynamic rescaling.
        gnorm_int_proj = otu.tree_norm(grads_int_proj)

        # --- 3b. KKT projected gradient norm in PHYSICAL space (for termination) ---
        # The internal-space norm is inflated by ``fscale`` during dynamic
        # rescaling and is unsuitable as a KKT measure. The physical-space
        # projected gradient norm goes to zero at any KKT point, by definition.
        grads_phys_proj = jax.tree.map(lambda p, g: jnp.where(p == 0, g, 0.0), pivot, grads)
        gnorm_proj = otu.tree_norm(grads_phys_proj)

        # --- 4. Dynamic Rescaling (TNC Logic) ---
        safe_gnorm = gnorm_int_proj + 1e-20
        should_rescale = (gnorm_int_proj > 1e-20) & (
            jnp.abs(jnp.log10(safe_gnorm)) > rescale_threshold
        )

        grads_int_proj = otu.tree_where(
            should_rescale, otu.tree_scale(1.0 / safe_gnorm, grads_int_proj), grads_int_proj
        )
        new_fscale = jnp.where(should_rescale, state.fscale / safe_gnorm, state.fscale)

        current_dir_state = otu.tree_where(
            should_rescale,
            _rescale_adam_state(state.direction_state, 1.0 / safe_gnorm),
            state.direction_state,
        )

        # --- 5. Compute Direction (pk) ---
        pk, new_dir_state = direction_solver.update(grads_int_proj, current_dir_state, params)

        # --- FIX 4: AdaBelief nu degeneration fallback ---
        pk = jax.tree.map(lambda p, pk: jnp.where(p == 0, pk, 0.0), state.pivot, pk)

        # --- 6. Step Limit (spe) ---
        pk_norm = otu.tree_norm(pk)
        ustpmax = state.stepmx / (pk_norm + 1e-20)
        spe = _compute_step_max(
            ustpmax, y_int, pk, pivot, state.lower, state.upper, state.xscale, state.offset
        )

        # --- 7. Line Search ---
        def internal_value_fn(y_candidate: PyTree[Float[Array, " P"]]) -> Scalar:
            x_candidate = otu.tree_add(otu.tree_mul(y_candidate, state.xscale), state.offset)
            x_candidate = jtu.tree_map(jnp.clip, x_candidate, state.lower, state.upper)
            return value_fn(x_candidate) * new_fscale

        ls_update_int, new_ls_state = linesearch_solver.update(
            pk,
            state.linesearch_state,
            y_int,
            value=value * new_fscale,
            grad=grads_int_proj,
            value_fn=internal_value_fn,
        )

        # --- 8. Step Clamping & Pivot Update ---
        ls_step_len = otu.tree_norm(ls_update_int)
        max_len = spe * pk_norm
        hit_wall = ls_step_len > max_len + 1e-10
        clamp_scale = jnp.where(hit_wall, max_len / (ls_step_len + 1e-20), 1.0)
        final_update_int = otu.tree_scale(clamp_scale, ls_update_int)

        final_pivot = lax.cond(
            hit_wall,
            lambda: _update_pivot_at_boundary(
                y_int, pk, pivot, state.lower, state.upper, state.xscale, state.offset, spe
            ),
            lambda: pivot,
        )

        # --- 9. Unscale Updates ---
        updates_phys = otu.tree_mul(final_update_int, state.xscale)

        new_last_release = jnp.where(constraints_released, state.count + 1, state.last_release_step)

        new_state = ActiveSetState(
            count=state.count + 1,
            pivot=final_pivot,
            xscale=state.xscale,
            offset=state.offset,
            lower=state.lower,
            upper=state.upper,
            fscale=new_fscale,
            stepmx=state.stepmx,
            max_release_k=state.max_release_k,
            direction_state=new_dir_state,
            linesearch_state=new_ls_state,
            constraints_released=constraints_released,
            last_release_step=new_last_release,
            best_f=jnp.minimum(state.best_f, value),
            f_val=value,
            prev_f=state.f_val,
            gnorm_proj=gnorm_proj,
        )

        return updates_phys, new_state

    # ExtraArgs (not plain GradientTransformation) so optax.chain forwards the
    # required value/value_fn kwargs when this is chained (e.g. with log_history).
    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


# =============================================================================
# Optimistix wrapper: KKT termination + cooldown
# =============================================================================


class ActiveSetMinimiser(optx.OptaxMinimiser):
    """Optimistix wrapper around an active-set optax transformation.

    Implements a robust termination protocol combining a KKT (projected
    gradient norm) test with a Cauchy y-space fallback (gated by a relaxed
    gradient-magnitude check to avoid false convergence on line-search
    stalls), plus a cooldown window after each constraint release to absorb
    transient spikes.

    References:
        Bertsekas (1982) — projected Newton for bound-constrained optimization
        Lin & Moré (1999) — Newton's method for large bound-constrained
    """

    cooldown_steps: int
    verbose_print: bool

    def __init__(
        self,
        optim,
        atol,
        rtol,
        cooldown_steps: int = 20,
        verbose_print: bool = False,
        **kwargs: Any,
    ):
        super().__init__(optim, atol=atol, rtol=rtol, **kwargs)
        self.cooldown_steps = cooldown_steps
        self.verbose_print = verbose_print

    def terminate(
        self,
        fn: Any,
        y: PyTree,
        args: PyTree,
        options: dict[str, Any],
        state: Any,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], optx.RESULTS]:
        del fn, args, options
        # opt_state is a chain tuple (ActiveSetState first) when a log_history
        # transform is attached; otherwise it is the ActiveSetState directly.
        opt = state.opt_state
        ast = opt if isinstance(opt, ActiveSetState) else opt[0]

        # Scale by best_f so termination is invariant to absolute objective magnitude.
        scale = jnp.maximum(1.0, jnp.abs(ast.best_f))

        # PRIMARY: KKT condition on the physical-space projected gradient.
        # Goes to zero at any true stationary point of the constrained problem.
        kkt_tol = self.atol + self.rtol * scale
        grad_converged = ast.gnorm_proj < kkt_tol

        # SECONDARY (anti-stall): Cauchy y-convergence is allowed to fire only
        # when the gradient is already "small enough" — using sqrt(tol) as a
        # relaxed threshold. This catches the numerical floor of the rescaling
        # pipeline (gnorm bottoms out at ~1e-7 because of fscale roundoff) but
        # cannot satisfy the strict KKT bound. Gated by ``state.terminate``
        # (optimistix Cauchy) to avoid firing on pure line-search stalls.
        relaxed_tol = jnp.sqrt(self.atol) + jnp.sqrt(self.rtol) * scale
        grad_small_enough = ast.gnorm_proj < relaxed_tol

        f_diff = jnp.abs(ast.f_val - ast.prev_f)  # reported for debug only
        converged = grad_converged | (state.terminate & grad_small_enough)

        # Cooldown: suppress termination during the window after a release,
        # because the transient may artificially satisfy convergence checks.
        steps_since_release = ast.count - ast.last_release_step
        in_cooldown = (ast.last_release_step >= 0) & (steps_since_release < self.cooldown_steps)
        override = ast.constraints_released | in_cooldown

        terminate = jnp.where(override, False, converged)

        if self.verbose_print:
            jax.debug.print(
                "step={s} | f={f:.4e} best_f={bf:.4e} gnorm={gn:.4e} f_diff={fd:.4e} "
                "scale={sc:.4e} | kkt={kkt} cauchy={cau} cooldown={cd} released={rel} "
                "=> terminate={t}",
                s=ast.count,
                f=ast.f_val,
                bf=ast.best_f,
                gn=ast.gnorm_proj,
                fd=f_diff,
                sc=scale,
                kkt=grad_converged,
                cau=state.terminate,
                cd=in_cooldown,
                rel=ast.constraints_released,
                t=terminate,
            )

        return terminate, optx.RESULTS.successful


# =============================================================================
# ADABK solver factory
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
    **kwargs: Any,
) -> optx.BestSoFarMinimiser:
    """Build an ADABK active-set minimiser wrapped in ``optx.BestSoFarMinimiser``.

    Recognised kwargs:

    * ``learning_rate`` (float, default 1.0) — AdaBelief learning rate.
    * ``linesearch`` (str, default ``"zoom"``) — ``"zoom"`` or ``"backtracking"``.
    * ``max_constraints_to_release`` (int|float|None) — overrides the
      ``ADABK{N}`` prefix parsing. Float is interpreted as a fraction of total
      params, int as an absolute count.

    References:
        Kabalan et al. (2025), arXiv:2604.08463 — ADABK / AdaTopK active set
        Zhuang et al. (2020), NeurIPS — AdaBelief Optimizer
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
                verbose_print=verbose_print,
                **kwargs,
            ),
            atol=atol,
            rtol=rtol,
            cooldown_steps=cooldown,
            verbose_print=verbose_print,
        )
    )
