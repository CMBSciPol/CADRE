"""Tests for the scipy solvers of ``minimize``: tolerances and options."""

from __future__ import annotations

import threading
import warnings

import jax
import jax.numpy as jnp
import pytest
from cadre import minimize, scipy_minimize

pytest.importorskip("jaxopt")

jax.config.update("jax_enable_x64", True)

TARGET = jnp.array([1.5, -2.0, 0.5])
X0 = jnp.zeros(3)


def quadratic(x):
    return jnp.sum(jnp.array([1.0, 3.0, 10.0]) * (x - TARGET) ** 2)


def run(**kwargs):
    return minimize(quadratic, X0, solver_name="scipy_tnc", max_iter=200, **kwargs)


def test_converges():
    params, state = run(rtol=1e-12, atol=1e-12)
    assert jnp.allclose(params, TARGET, atol=1e-6)
    assert int(state.iter_num) > 0


def test_tolerances_reach_scipy():
    """Loose tolerances stop the solver earlier than tight ones."""
    _, loose = run(rtol=1e-1, atol=1e-1)
    _, tight = run(rtol=1e-12, atol=1e-12)
    assert int(loose.iter_num) < int(tight.iter_num)


def test_options_reach_scipy():
    """An unknown option is reported by scipy, so the options are handed to it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(options={"not_an_option": 1})
    assert any("not_an_option" in str(w.message) for w in caught)


def test_explicit_options_win_over_tolerances():
    _, loose = run(rtol=1e-1, atol=1e-1)
    _, explicit = run(rtol=1e-12, atol=1e-12, options={"ftol": 1e-1, "gtol": 1e-1, "xtol": 1e-1})
    assert int(explicit.iter_num) == int(loose.iter_num)


def test_options_not_modified():
    options = {"maxCGit": 5}
    run(options=options)
    assert options == {"maxCGit": 5}


def test_options_under_vmap():
    """The callback runs once per batch member: each run gets its own options."""
    x0s = jnp.stack([X0, X0 + 1.0])
    states = jax.vmap(
        lambda x0: scipy_minimize(quadratic, x0, method="tnc", options={"ftol": 1e-12})
    )(x0s)
    assert jnp.allclose(states.params, TARGET, atol=1e-5)


# =============================================================================
# host_callback=False: the solver on the calling thread
# =============================================================================


def test_calling_thread_matches_host_callback():
    params, state = run(rtol=1e-12, atol=1e-12)
    params_ct, state_ct = run(rtol=1e-12, atol=1e-12, host_callback=False)
    assert jnp.allclose(params_ct, params, rtol=0, atol=1e-10)
    assert int(state_ct.iter_num) == int(state.iter_num)
    assert state_ct.solver_state.fun_val.dtype == state.solver_state.fun_val.dtype


def test_calling_thread_forwards_fn_kwargs():
    def objective(x, scale):
        return scale * quadratic(x)

    params, _ = minimize(
        objective, X0, solver_name="scipy_tnc", max_iter=200, rtol=1e-12, atol=1e-12,
        host_callback=False, scale=jnp.array(10.0),
    )  # fmt: skip
    assert jnp.allclose(params, TARGET, atol=1e-6)


def test_calling_thread_sees_the_mesh():
    """The objective is traced on the calling thread, where the (thread-local) mesh is set."""
    seen = []

    def objective(x):
        seen.append((threading.get_ident(), jax.sharding.get_abstract_mesh().axis_names))
        return quadratic(x)

    with jax.set_mesh(jax.make_mesh((1,), ("x",))):
        minimize(objective, X0, solver_name="scipy_tnc", max_iter=50, host_callback=False)
    assert seen and all(entry == (threading.get_ident(), ("x",)) for entry in seen)


def test_calling_thread_rejects_transformations():
    def solve(x0):
        return minimize(quadratic, x0, solver_name="scipy_tnc", host_callback=False)[0]

    with pytest.raises(ValueError, match="host_callback"):
        jax.jit(solve)(X0)
