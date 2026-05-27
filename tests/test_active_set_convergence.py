"""Convergence tests for ``active_set`` on unconstrained-minimum problems.

The four objective functions used here all have their global minimum strictly
inside the chosen box, so ``active_set`` should reach the minimum without ever
activating a constraint and should perform at least as well as the underlying
Adam + backtracking-linesearch chain.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from cadre.active_set import active_set

jax.config.update("jax_numpy_dtype_promotion", "standard")


# =============================================================================
# Objective functions
# =============================================================================


def rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def sphere(x):
    return jnp.sum(x**2)


def beale(x):
    return (
        (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    )


# =============================================================================
# Driver
# =============================================================================


MAX_STEPS = 2000


def _run(solver, params, value_and_grad_fn, max_steps, lower, upper):
    state = solver.init(params)
    value_fn = lambda p: value_and_grad_fn(p)[0]

    @eqx.filter_jit
    def step(params, state):
        val, grad = value_and_grad_fn(params)
        updates, new_state = solver.update(
            grad, state, params, value=val, grad=grad, value_fn=value_fn
        )
        new_params = optax.apply_updates(params, updates)
        new_params = jnp.minimum(jnp.maximum(new_params, lower), upper)
        return new_params, new_state, val

    losses = []
    for _ in range(max_steps):
        params, state, val = step(params, state)
        jax.block_until_ready(params)
        losses.append(float(val))
    return params, losses


# =============================================================================
# Parametrized convergence tests
# =============================================================================


@pytest.mark.parametrize(
    "name, fn, start, bounds_range, loss_tol",
    [
        ("sphere_10d", sphere, [3.0] * 10, 10.0, 1e-6),
        ("rosenbrock_2d", rosenbrock, [0.0, 0.0], 5.0, 1e-3),
        ("himmelblau_2d", himmelblau, [0.0, 0.0], 5.0, 1e-6),
        ("beale_2d", beale, [0.0, 0.0], 5.0, 1e-6),
    ],
)
def test_active_set_converges_at_least_as_well_as_baseline(name, fn, start, bounds_range, loss_tol):
    """active_set must reach the minimum and be no worse than the unwrapped chain."""
    del name  # used only for test-id readability

    params = jnp.array(start)
    lower = -jnp.ones_like(params) * bounds_range
    upper = jnp.ones_like(params) * bounds_range
    vg = jax.value_and_grad(fn)

    direction = optax.adam(learning_rate=0.1)
    linesearch = optax.scale_by_backtracking_linesearch(max_backtracking_steps=5)

    baseline = optax.chain(direction, linesearch)
    active = active_set(direction, linesearch, lower=lower, upper=upper)

    _, loss_baseline = _run(baseline, params, vg, MAX_STEPS, lower, upper)
    _, loss_active = _run(active, params, vg, MAX_STEPS, lower, upper)

    assert (
        loss_active[-1] < loss_tol
    ), f"active_set final loss {loss_active[-1]:.3e} did not reach tol {loss_tol:.0e}"
    assert loss_active[-1] <= loss_baseline[-1] + 1e-6, (
        f"active_set final loss {loss_active[-1]:.3e} worse than "
        f"baseline {loss_baseline[-1]:.3e}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
