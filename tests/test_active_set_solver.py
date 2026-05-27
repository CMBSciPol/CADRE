"""Tests for the CADRE active-set solver family (ADABK).

References:
    Kabalan et al. (2025), arXiv:2604.08463 — AdaTopK active-set method
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from cadre import minimize

jax.config.update("jax_enable_x64", True)


# =============================================================================
# Problem definitions
# =============================================================================


def _noisy_rosenbrock_factory(seed: int, noise_scale: float = 1.5):
    """Return a noisy 10-D Rosenbrock with a fixed-seed additive Gaussian noise.

    The noise is constant in x (only depends on the fixed seed), so the
    gradient is unchanged but the function value has a stochastic offset
    that exercises the optimizer's tolerance to a non-zero noise floor.
    """
    noise_key = jax.random.PRNGKey(seed)
    noise = float(jax.random.normal(noise_key) * noise_scale)

    def f(x):
        return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2) + noise

    return f


N_DIMS = 10
X0 = jnp.array([-1.2, 1.0] * (N_DIMS // 2))


# =============================================================================
# Smoke test — constrained 2-D Rosenbrock
# =============================================================================


def test_rosenbrock_constrained():
    """Active set hits the upper bound when the unconstrained min lies past it.

    Standard 2-D Rosenbrock has its minimum at (1, 1). With ``a <= 0.5``, the
    solver should stop with ``a == 0.5`` (active upper bound) and ``b == 0.25``
    (the conditional minimum given a=0.5).
    """
    params = {"a": jnp.array([0.0]), "b": jnp.array([0.0])}
    lower = {"a": jnp.array([-jnp.inf]), "b": jnp.array([-jnp.inf])}
    upper = {"a": jnp.array([0.5]), "b": jnp.array([jnp.inf])}

    def rosen(p):
        return jnp.sum((1.0 - p["a"]) ** 2 + 100.0 * (p["b"] - p["a"] ** 2) ** 2)

    final_params, _state = minimize(
        rosen,
        params,
        solver_name="ADABK0",
        max_iter=5000,
        lower_bound=lower,
        upper_bound=upper,
        atol=1e-12,
        rtol=1e-12,
    )

    assert jnp.allclose(final_params["a"], 0.5, atol=2e-2), final_params
    assert jnp.allclose(final_params["b"], 0.25, atol=2e-2), final_params


# =============================================================================
# KKT termination — ADABK0 should stop early when stationary
# =============================================================================


def test_adabk0_terminates_via_kkt_on_quadratic():
    """A clean quadratic has a single stationary point with a vanishing gradient.

    KKT termination must fire well before ``max_iter`` (it took 2000 iters
    before the KKT criterion was added).
    """
    target = jnp.array([0.3, -0.7, 0.5, 0.1, 0.0])

    def quad(x):
        return jnp.sum((x - target) ** 2)

    x0 = jnp.zeros_like(target) + 0.1
    _, state = minimize(
        quad,
        x0,
        solver_name="ADABK0",
        max_iter=500,
        lower_bound=jnp.full_like(target, -5.0),
        upper_bound=jnp.full_like(target, 5.0),
    )
    assert (
        int(state.iter_num) < 300
    ), f"ADABK0 should terminate via KKT in < 300 iters; got {int(state.iter_num)}"
    assert float(state.best_loss) < 1e-8


# =============================================================================
# Multi-seed comparison — ADABK0 vs optax_lbfgs on noisy Rosenbrock
# =============================================================================


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adabk0_reaches_same_minimum_as_lbfgs_noisy(seed):
    """ADABK0 (first-order) must still reach the same loss as L-BFGS, given enough iters."""
    f = _noisy_rosenbrock_factory(seed)
    lo = jnp.full_like(X0, -5.0)
    up = jnp.full_like(X0, 10.0)
    _, st_adabk = minimize(
        f,
        X0,
        solver_name="ADABK0",
        max_iter=3000,
        lower_bound=lo,
        upper_bound=up,
    )
    _, st_lbfgs = minimize(
        f,
        X0,
        solver_name="optax_lbfgs",
        max_iter=3000,
        lower_bound=lo,
        upper_bound=up,
    )
    assert float(st_adabk.best_loss) <= float(st_lbfgs.best_loss) + 1e-3, (
        f"seed={seed}: ADABK0 loss {float(st_adabk.best_loss):.4e} should be "
        f"≤ optax_lbfgs loss {float(st_lbfgs.best_loss):.4e}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
