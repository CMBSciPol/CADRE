"""AdaTopK: projected AdaBelief with a Top-K active set, as an optimistix minimiser.

References:
    Kabalan et al. (2026), arXiv:2604.08463 — AdaTopK active-set method
    Zhuang et al. (2020), NeurIPS — AdaBelief optimizer
    Bertsekas (1982), SIAM J. Control Optim. 20, 221 — projected Armijo rule

The parameters are mapped to internal coordinates ``z = (x - offset) / scale``, in which each
bounded parameter lives in ``[-1/2, 1/2]``. AdaTopK is therefore self-conditioned, and
``cadre.minimize`` ignores ``precondition=True`` for it.

One iteration (:meth:`AdaTopK.step`):

1. Release: among the pinned parameters whose gradient points into the box, release the
   ``k`` with the largest score ``pivot * g`` (``k = 1`` for ``ADABK0``).
2. Direction: AdaBelief on the gradient of the free parameters. If the AdaBelief direction
   is not a descent direction (its momentum points uphill), steepest descent of the same
   length is used for that step.
3. Step: projected path ``z(a) = clip(z + a p)``, coordinate by coordinate, so that a
   parameter close to its bound does not limit the step of the others.
4. Line search: Armijo backtracking along the projected path. Each trial evaluates the value
   and the gradient, so the accepted trial provides the gradient of the next iteration. The
   first trial is twice the last accepted step.
5. Pinning: the parameters that landed on a bound while moving outwards are pinned.
6. Restart or stop: when the accepted step falls below ``restart_alpha`` or the line search
   fails, the AdaBelief moments and the step length are reset if the best value improved by
   more than ``ftol * max(1, |f|)`` since the last restart; otherwise the solver stops.

Termination (:meth:`AdaTopK.terminate`): the KKT residual (projected gradient in physical
units) is ``<= atol + rtol * KKT(x0)``, or step 6 stops the solver. Termination is suppressed
for ``cooldown`` iterations after a release. The iteration cap is the ``max_steps`` argument
of :func:`optimistix.minimise`.

Defaults: heavy momentum (``b1 = 0.99``) carries the iterate along the flat, correlated
valleys of the spectral likelihood (e.g. dust temperature and index in low-SNR patches);
a short second-moment memory (``b2 = 0.9``) keeps the per-parameter scales current.

Example::

    solver = AdaTopK(rtol=0.0, atol=1e-10, lower=lower, upper=upper)
    sol = optx.minimise(fn, solver, x0, args, max_steps=10_000, throw=False)
    sol.value  # best parameters found
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

_PIN_TOL = 1e-12  # internal units (the box has unit width)


def release_count(solver_name: str, n_params: int, override: int | float | None = None) -> int:
    """Number of constraints released per iteration (``ADABK{N}`` → fraction ``N / 10``)."""
    value = override
    if value is None and solver_name.startswith("ADABK") and len(solver_name) > 5:
        value = int(solver_name[5:]) * 0.1
    if value is None:
        return 1
    k = int(value * n_params) if isinstance(value, float) else int(value)
    return max(1, min(k, n_params))


def _kkt_residual(g_int: Array, scale: Array, pivot: Array) -> Array:
    """Norm of the projected gradient in physical units (zero at a KKT point)."""
    g = g_int / scale
    r = jnp.where(
        pivot == 0,
        g,
        jnp.where(
            pivot == -1, jnp.minimum(g, 0.0), jnp.where(pivot == 1, jnp.maximum(g, 0.0), 0.0)
        ),
    )
    return jnp.linalg.norm(r)


def _tree_where(pred: Array, a: Any, b: Any) -> Any:
    return jax.tree.map(lambda u, v: jnp.where(pred, u, v), a, b)


class _Box(eqx.Module):
    """Map between the flat physical parameters ``x`` and the internal coordinates ``z``."""

    lower: Array  # physical lower bounds
    scale: Array
    offset: Array
    z_lower: Array
    z_upper: Array
    fixed: Array  # lower == upper: the parameter stays at its bound

    def to_x(self, z: Array) -> Array:
        return jnp.where(self.fixed, self.lower, z * self.scale + self.offset)


class AdaTopKState(eqx.Module):
    """State of :class:`AdaTopK`. ``z`` and ``best_z`` are internal coordinates (flat)."""

    box: _Box
    z: Array
    f: Array  # objective at z
    g: Array  # internal-space gradient at z
    aux: Any  # auxiliary output of fn at z
    pivot: Array  # 0 free, -1 lower bound, +1 upper bound, 2 fixed
    opt_state: optax.OptState  # AdaBelief state
    alpha: Array  # last accepted step length
    best_loss: Array
    best_z: Array
    best_aux: Any
    kkt: Array  # KKT residual at z, physical units
    kkt0: Array  # KKT residual at the start
    iter_num: Array
    last_release: Array  # iteration of the last release
    restart_f: Array  # best value at the last restart
    exhausted: Array  # step 6 stopped the solver
    n_evals: Array
    n_safeguard: Array  # steps where steepest descent replaced AdaBelief
    n_ls_fail: Array
    n_restart: Array
    history: Array  # (2, history_length): step norm, objective before the step


class _LineSearch(NamedTuple):
    k: Array
    alpha: Array
    z: Array
    f: Array
    g: Array
    aux: Any
    ok: Array


def _value_and_grad(fn: Callable, args: PyTree, unravel: Callable, box: _Box, z: Array):
    """Objective, internal-space gradient and auxiliary output at ``z``."""

    def objective(x: Array):
        return fn(unravel(x), args)

    (f, aux), grad_x = jax.value_and_grad(objective, has_aux=True)(box.to_x(z))
    return f, grad_x * box.scale, aux


def _flat_bound(bound: Any, fill: float, n: int, dtype: Any) -> Array:
    if bound is None:
        return jnp.full(n, fill, dtype)
    return jnp.broadcast_to(ravel_pytree(bound)[0].astype(dtype), (n,))


class AdaTopK(optx.AbstractMinimiser):
    """Projected AdaBelief with a Top-K active set (algorithm in the module docstring).

    ``lower`` and ``upper`` are pytrees matching the parameters, or scalars; ``None`` means
    unbounded. ``solver_name`` ``ADABK{N}`` releases a fraction ``N / 10`` of the parameters
    per iteration (``ADABK0``: one), unless ``max_constraints_to_release`` is set (an int is a
    count, a float a fraction). ``history_length > 0`` records the step norm (internal units)
    and the objective before the step for the first ``history_length`` iterations.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Array] = optx.max_norm  # required by optimistix, unused
    lower: Any = None
    upper: Any = None
    solver_name: str = eqx.field(static=True, default="ADABK0")
    max_constraints_to_release: int | float | None = eqx.field(static=True, default=None)
    cooldown: int = eqx.field(static=True, default=20)
    ftol: float = 1e-15
    max_linesearch_steps: int = eqx.field(static=True, default=30)
    armijo: float = 1e-4
    backtrack: float = 0.5
    initial_step: float = 1e-2
    restart_alpha: float = 1e-10
    b1: float = 0.99
    b2: float = 0.9
    eps: float = 1e-16
    history_length: int = eqx.field(static=True, default=0)
    verbose_print: bool = eqx.field(static=True, default=False)

    def _adabelief(self) -> optax.GradientTransformation:
        return optax.adabelief(
            learning_rate=1.0, b1=self.b1, b2=self.b2, eps=self.eps, eps_root=self.eps
        )

    def init(
        self,
        fn: Callable,
        y: PyTree,
        args: PyTree,
        options: dict[str, Any],
        f_struct: Any,
        aux_struct: Any,
        tags: frozenset[object],
    ) -> AdaTopKState:
        del options, f_struct, aux_struct, tags
        x0, unravel = ravel_pytree(y)
        dtype, n = x0.dtype, x0.size
        lo = _flat_bound(self.lower, -jnp.inf, n, dtype)
        up = _flat_bound(self.upper, jnp.inf, n, dtype)
        bounded = jnp.isfinite(lo) & jnp.isfinite(up)
        fixed = bounded & (up <= lo)
        scale = jnp.where(bounded & ~fixed, up - lo, 1.0 + jnp.abs(x0))
        offset = jnp.where(bounded, 0.5 * (lo + up), x0)
        box = _Box(
            lower=lo,
            scale=scale,
            offset=offset,
            z_lower=(lo - offset) / scale,
            z_upper=(up - offset) / scale,
            fixed=fixed,
        )
        z0 = jnp.clip((x0 - offset) / scale, box.z_lower, box.z_upper)
        pivot0 = jnp.where(
            fixed,
            2,
            jnp.where(
                z0 <= box.z_lower + _PIN_TOL, -1, jnp.where(z0 >= box.z_upper - _PIN_TOL, 1, 0)
            ),
        ).astype(jnp.int32)
        f0, g0, aux0 = _value_and_grad(fn, args, unravel, box, z0)
        kkt0 = _kkt_residual(g0, scale, pivot0)
        zero = jnp.array(0, jnp.int32)
        return AdaTopKState(
            box=box,
            z=z0,
            f=f0,
            g=g0,
            aux=aux0,
            pivot=pivot0,
            opt_state=self._adabelief().init(z0),
            alpha=jnp.array(self.initial_step, dtype),
            best_loss=f0,
            best_z=z0,
            best_aux=aux0,
            kkt=kkt0,
            kkt0=kkt0,
            iter_num=zero,
            last_release=jnp.array(-(10**9), jnp.int32),
            restart_f=f0,
            exhausted=jnp.array(False),
            n_evals=jnp.array(1, jnp.int32),
            n_safeguard=zero,
            n_ls_fail=zero,
            n_restart=zero,
            history=jnp.zeros((2, max(self.history_length, 1)), dtype),
        )

    def step(
        self,
        fn: Callable,
        y: PyTree,
        args: PyTree,
        options: dict[str, Any],
        state: AdaTopKState,
        tags: frozenset[object],
    ) -> tuple[PyTree, AdaTopKState, Any]:
        del options, tags
        s, box = state, state.box
        unravel = ravel_pytree(y)[1]
        n = s.z.size
        opt = self._adabelief()

        # 1. Release the Top-K pinned parameters whose gradient points into the box.
        k = release_count(self.solver_name, n, self.max_constraints_to_release)
        score = jnp.where(jnp.abs(s.pivot) == 1, s.pivot * s.g, -jnp.inf)
        top_val, top_idx = jax.lax.top_k(score, k)
        release = jnp.zeros(n, bool).at[top_idx].set(top_val > 0)
        pivot = jnp.where(release, 0, s.pivot)
        released = jnp.any(release)
        free = pivot == 0
        g_free = jnp.where(free, s.g, 0.0)

        # 2. AdaBelief direction with descent safeguard.
        p, opt_state = opt.update(g_free, s.opt_state, s.z)
        p = jnp.where(free, p, 0.0)
        p_norm = jnp.linalg.norm(p)
        g_norm = jnp.linalg.norm(g_free)
        ascent = jnp.vdot(p, g_free) >= 0.0
        ratio = jnp.where(
            (p_norm > 0) & (g_norm > 0), p_norm / jnp.where(g_norm > 0, g_norm, 1.0), 1.0
        )
        p = jnp.where(ascent, -g_free * ratio, p)

        # 3-4. Projected Armijo backtracking, value and gradient at each trial.
        alpha0 = jnp.minimum(1.0, 2.0 * s.alpha)

        def ls_cond(c: _LineSearch) -> Array:
            return (~c.ok) & (c.k < self.max_linesearch_steps)

        def ls_body(c: _LineSearch) -> _LineSearch:
            a = alpha0 * self.backtrack**c.k
            z_t = jnp.clip(s.z + a * p, box.z_lower, box.z_upper)
            f_t, g_t, aux_t = _value_and_grad(fn, args, unravel, box, z_t)
            decrease = jnp.minimum(jnp.vdot(s.g, z_t - s.z), 0.0)
            ok = jnp.isfinite(f_t) & (f_t <= s.f + self.armijo * decrease)
            return _LineSearch(c.k + 1, a, z_t, f_t, g_t, aux_t, ok)

        ls = jax.lax.while_loop(
            ls_cond,
            ls_body,
            _LineSearch(jnp.array(0, jnp.int32), alpha0, s.z, s.f, s.g, s.aux, jnp.array(False)),
        )
        z = jnp.where(ls.ok, ls.z, s.z)
        f = jnp.where(ls.ok, ls.f, s.f)
        g = jnp.where(ls.ok, ls.g, s.g)
        aux = _tree_where(ls.ok, ls.aux, s.aux)

        # 5. Pin the parameters that landed on a bound while moving outwards.
        hit_lo = ls.ok & free & (p < 0) & (z <= box.z_lower + _PIN_TOL)
        hit_up = ls.ok & free & (p > 0) & (z >= box.z_upper - _PIN_TOL)
        pivot = jnp.where(hit_lo, -1, jnp.where(hit_up, 1, pivot))

        # 6. Restart or stop. A collapsed step (or a failed line search) means either that the
        #    AdaBelief moments are stale (a second moment inflated by early gradient swings
        #    freezes some coordinates while converged ones take near-unit normalised steps), or
        #    that the objective cannot be decreased further at this precision. If the best value
        #    improved since the last restart, reset moments and step; otherwise stop.
        collapsed = (~ls.ok) | (ls.alpha < self.restart_alpha)
        best_now = jnp.minimum(s.best_loss, f)
        improved = (s.restart_f - best_now) > self.ftol * jnp.maximum(1.0, jnp.abs(best_now))
        restart = collapsed & improved
        exhausted = collapsed & ~improved
        restart_f = jnp.where(restart, best_now, s.restart_f)
        opt_state = _tree_where(restart, opt.init(z), opt_state)
        # A failed line search keeps the last accepted step, not its last rejected trial.
        alpha = jnp.where(
            restart,
            jnp.asarray(self.initial_step, s.alpha.dtype),
            jnp.where(ls.ok, ls.alpha, s.alpha),
        )

        # Bookkeeping.
        it = s.iter_num + 1
        better = f < s.best_loss
        history = s.history
        if self.history_length > 0:
            history = history.at[:, s.iter_num].set(jnp.stack([jnp.linalg.norm(z - s.z), s.f]))
        new_state = AdaTopKState(
            box=box,
            z=z,
            f=f,
            g=g,
            aux=aux,
            pivot=pivot,
            opt_state=opt_state,
            alpha=alpha,
            best_loss=jnp.where(better, f, s.best_loss),
            best_z=jnp.where(better, z, s.best_z),
            best_aux=_tree_where(better, aux, s.best_aux),
            kkt=_kkt_residual(g, box.scale, pivot),
            kkt0=s.kkt0,
            iter_num=it,
            last_release=jnp.where(released, it, s.last_release),
            restart_f=restart_f,
            exhausted=exhausted,
            n_evals=s.n_evals + ls.k,
            n_safeguard=s.n_safeguard + ascent.astype(jnp.int32),
            n_ls_fail=s.n_ls_fail + (~ls.ok).astype(jnp.int32),
            n_restart=s.n_restart + restart.astype(jnp.int32),
            history=history,
        )
        if self.verbose_print:
            jax.debug.print(
                "it={it} f={f:.10e} kkt={kkt:.3e} alpha={a:.2e} ls={ls} ok={ok} ascent={asc} "
                "free={nf} released={rel} restart={rs} exhausted={ex}",
                it=it,
                f=f,
                kkt=new_state.kkt,
                a=ls.alpha,
                ls=ls.k,
                ok=ls.ok,
                asc=ascent,
                nf=jnp.sum(pivot == 0),
                rel=released,
                rs=restart,
                ex=exhausted,
            )
        # optimistix convention: the auxiliary output returned is the one at the input ``y``.
        return unravel(box.to_x(z)), new_state, s.aux

    def terminate(
        self,
        fn: Callable,
        y: PyTree,
        args: PyTree,
        options: dict[str, Any],
        state: AdaTopKState,
        tags: frozenset[object],
    ) -> tuple[Array, optx.RESULTS]:
        del fn, y, args, options, tags
        converged = (state.kkt <= self.atol + self.rtol * state.kkt0) | state.exhausted
        in_cooldown = (state.iter_num - state.last_release) < self.cooldown
        return converged & ~in_cooldown, optx.RESULTS.successful

    def postprocess(
        self,
        fn: Callable,
        y: PyTree,
        aux: Any,
        args: PyTree,
        options: dict[str, Any],
        state: AdaTopKState,
        tags: frozenset[object],
        result: optx.RESULTS,
    ) -> tuple[PyTree, Any, dict[str, Any]]:
        del fn, aux, args, options, tags, result
        unravel = ravel_pytree(y)[1]
        return unravel(state.box.to_x(state.best_z)), state.best_aux, {}
