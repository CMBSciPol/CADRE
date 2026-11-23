"""Tests for AdaTopK (``cadre.adatopk``), the ``ADABK{N}`` solvers.

References:
    Kabalan et al. (2026), arXiv:2604.08463 — AdaTopK active-set method
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from cadre import AdaTopK, get_solver, minimize
from cadre.adatopk import release_count

jax.config.update("jax_enable_x64", True)

# Separable weighted quadratic: the minimum in the box is clip(TARGET, LOWER, UPPER).
# Coordinates 0 and 3 end on the lower and upper bound.
TARGET = jnp.array([-2.0, 0.3, 1.7, 5.0, -0.4, 0.9])
WEIGHTS = jnp.array([1.0, 10.0, 0.1, 3.0, 100.0, 1.0])
LOWER = jnp.full(6, -1.0)
UPPER = jnp.full(6, 2.0)
X0 = jnp.zeros(6)
SOLUTION = jnp.clip(TARGET, LOWER, UPPER)


def quadratic(x, target):
    return jnp.sum(WEIGHTS * (x - target) ** 2)


def run(x0=X0, lower=LOWER, upper=UPPER, solver_name="ADABK0", fn=quadratic, **kwargs):
    kwargs.setdefault("max_iter", 5000)
    return minimize(
        fn,
        x0,
        solver_name=solver_name,
        lower_bound=lower,
        upper_bound=upper,
        atol=1e-10,
        rtol=0.0,
        progress=False,
        target=TARGET,
        **kwargs,
    )


def test_reaches_box_solution():
    params, state = run()
    np.testing.assert_allclose(params, SOLUTION, atol=1e-6)
    assert int(state.solver_state.pivot[0]) == -1
    assert int(state.solver_state.pivot[3]) == 1
    assert float(state.best_loss) == pytest.approx(float(quadratic(params, TARGET)), rel=1e-12)


def test_iterates_stay_in_box():
    seen = []

    def logged(x, target):
        jax.debug.callback(lambda v: seen.append(np.asarray(v)), x)
        return quadratic(x, target)

    start = jnp.array([3.0, -3.0, 0.0, 1.0, 5.0, -5.0])  # outside the box: clipped at init
    params, _ = run(x0=start, fn=logged, max_iter=200)
    points = np.stack(seen)
    assert len(points) > 1
    assert np.all(points >= np.asarray(LOWER)) and np.all(points <= np.asarray(UPPER))


def test_kkt_termination():
    solver = AdaTopK(rtol=0.0, atol=1e-6, lower=LOWER, upper=UPPER)
    sol = optx.minimise(quadratic, solver, X0, TARGET, max_steps=5000, throw=False)
    assert bool(sol.result == optx.RESULTS.successful)
    assert int(sol.stats["num_steps"]) < 5000
    assert float(sol.state.kkt) <= 1e-6 or bool(sol.state.exhausted)


def test_max_steps():
    _, state = run(max_iter=5)
    assert int(state.iter_num) == 5
    solver = AdaTopK(rtol=0.0, atol=0.0, lower=LOWER, upper=UPPER)
    sol = optx.minimise(quadratic, solver, X0, TARGET, max_steps=5, throw=False)
    assert bool(sol.result == optx.RESULTS.nonlinear_max_steps_reached)


def test_fixed_parameter():
    lower = LOWER.at[2].set(0.5)
    upper = UPPER.at[2].set(0.5)
    params, _ = run(lower=lower, upper=upper)
    assert float(params[2]) == 0.5
    np.testing.assert_allclose(np.delete(params, 2), np.delete(SOLUTION, 2), atol=1e-6)


def test_pytree_and_scalar_bounds():
    def split(x, target):
        return quadratic(jnp.concatenate([x["a"], x["b"]]), target)

    x0 = {"a": X0[:4], "b": X0[4:]}
    params, _ = run(
        x0=x0,
        fn=split,
        lower={"a": LOWER[:4], "b": LOWER[4:]},
        upper={"a": UPPER[:4], "b": UPPER[4:]},
    )
    np.testing.assert_allclose(jnp.concatenate([params["a"], params["b"]]), SOLUTION, atol=1e-6)
    params_scalar, _ = run(lower=-1.0, upper=2.0)
    np.testing.assert_allclose(params_scalar, SOLUTION, atol=1e-6)


def test_jit_vmap_over_starts():
    starts = jnp.stack([X0, X0 + 0.5, X0 - 0.5, jnp.full(6, 1.0)])
    batched = jax.jit(jax.vmap(lambda x0: run(x0=x0)[0]))(starts)
    for i in range(starts.shape[0]):
        np.testing.assert_allclose(batched[i], run(x0=starts[i])[0], atol=1e-6)
        np.testing.assert_allclose(batched[i], SOLUTION, atol=1e-6)


def test_precondition_is_ignored():
    params, state = run(precondition=True)
    params_ref, state_ref = run()
    np.testing.assert_array_equal(params, params_ref)
    assert int(state.iter_num) == int(state_ref.iter_num)


def test_record_history():
    _, state = run(max_iter=50, record_history=True)
    history = state.history
    assert history is not None
    assert history.shape == (2, 50)
    assert float(history[1, 0]) == pytest.approx(float(quadratic(X0, TARGET)))
    assert float(history[0, 0]) > 0.0


@pytest.mark.parametrize(
    ("name", "n", "override", "expected"),
    [
        ("ADABK0", 100, None, 1),
        ("ADABK", 100, None, 1),
        ("ADABK5", 100, None, 50),
        ("ADABK0", 100, 7, 7),
        ("ADABK0", 100, 0.2, 20),
        ("ADABK9", 3, None, 2),
    ],
)
def test_release_count(name, n, override, expected):
    assert release_count(name, n, override) == expected


def test_get_solver_adabk():
    solver, kind = get_solver("ADABK3", lower=LOWER, upper=UPPER, cooldown=7)
    assert kind == "optimistix" and isinstance(solver, AdaTopK)
    assert solver.solver_name == "ADABK3" and solver.cooldown == 7
    assert solver.max_linesearch_steps == 30
    solver, _ = get_solver("ADABK0", max_linesearch_steps=12)
    assert solver.max_linesearch_steps == 12


@pytest.mark.parametrize("name", ["ADABKx", "ADABK-3", "ADABK1.5"])
def test_get_solver_rejects_malformed_adabk_name(name):
    with pytest.raises(ValueError, match="AdaTopK solver name"):
        get_solver(name)
