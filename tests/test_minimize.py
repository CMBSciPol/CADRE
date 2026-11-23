"""Tests for ``cadre.minimize`` and ``cadre.get_solver`` outside AdaTopK."""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadre import minimize

jax.config.update("jax_enable_x64", True)


def quadratic(x):
    return jnp.sum((x - 0.5) ** 2)


@pytest.mark.parametrize("alias", ["zoom", "backtrack"])
def test_legacy_lbfgs_aliases(alias):
    params, _ = minimize(quadratic, jnp.zeros(3), solver_name=alias, max_iter=200, progress=False)
    np.testing.assert_allclose(params, 0.5, atol=1e-6)


def test_scipy_receives_tolerances_and_keeps_caller_options(monkeypatch):
    module = importlib.import_module("cadre.minimize")
    captured = {}

    def fake_scipy_minimize(
        fn, init_params, lower_bound, upper_bound, method, maxiter, solver_options, **fn_kwargs
    ):
        captured.update(method=method, solver_options=solver_options)
        return module.ScipyMinimizeState(
            params=init_params,
            fun_val=jnp.zeros(()),
            success=jnp.array(True),
            iter_num=jnp.array(0),
        )

    monkeypatch.setattr(module, "scipy_minimize", fake_scipy_minimize)
    options = {"maxCGit": 5, "cooldown": 10}
    minimize(
        quadratic,
        jnp.zeros(3),
        solver_name="scipy_tnc",
        atol=1e-12,
        rtol=1e-9,
        options=options,
        progress=False,
    )
    assert captured["method"] == "tnc"
    assert captured["solver_options"] == {"maxCGit": 5, "ftol": 1e-12, "gtol": 1e-9, "xtol": 1e-12}
    assert options == {"maxCGit": 5, "cooldown": 10}


def test_scipy_tnc_solves_bounded_quadratic():
    pytest.importorskip("jaxopt")
    params, _ = minimize(
        quadratic,
        jnp.zeros(3),
        solver_name="scipy_tnc",
        lower_bound=-jnp.ones(3),
        upper_bound=jnp.ones(3),
        atol=1e-12,
        rtol=1e-12,
        progress=False,
    )
    np.testing.assert_allclose(params, 0.5, atol=1e-5)
