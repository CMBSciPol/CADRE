"""Tests for the scipy solvers of ``minimize``: tolerances and options."""

from __future__ import annotations

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
