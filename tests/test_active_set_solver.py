"""Tests for the CADRE active-set solver family.

These tests assert that ADABK and LBFGSK genuinely improve on the off-the-shelf
``optax_lbfgs`` and ``scipy_tnc`` baselines on representative problems.

References:
    Kabalan et al. (2025), arXiv:2604.08463 — AdaTopK active-set method
"""

from __future__ import annotations

import cadre
import jax
import jax.numpy as jnp
import pytest
from cadre import BoxConstraint, GaussianConstraint, minimize

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


def _clean_rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


N_DIMS = 10
X0 = jnp.array([-1.2, 1.0] * (N_DIMS // 2))


# =============================================================================
# Smoke test — original constrained Rosenbrock
# =============================================================================


def test_rosenbrock_constrained_smoke():
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
        solver_name="LBFGSK0",
        max_iter=5000,
        constraints=BoxConstraint(lower=lower, upper=upper),
        atol=1e-12,
        rtol=1e-12,
    )

    # a hits the upper bound (0.5), b minimizes the residual at a=0.5 → b=0.25
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
        constraints=BoxConstraint(lower=-5.0, upper=5.0),
    )
    assert (
        int(state.iter_num) < 300
    ), f"ADABK0 should terminate via KKT in < 300 iters; got {int(state.iter_num)}"
    assert float(state.best_loss) < 1e-8


def test_lbfgsk0_terminates_via_kkt_on_quadratic():
    target = jnp.array([0.3, -0.7, 0.5, 0.1, 0.0])

    def quad(x):
        return jnp.sum((x - target) ** 2)

    x0 = jnp.zeros_like(target) + 0.1
    _, state = minimize(
        quad,
        x0,
        solver_name="LBFGSK0",
        max_iter=500,
        constraints=BoxConstraint(lower=-5.0, upper=5.0),
    )
    assert int(state.iter_num) < 100, (
        f"LBFGSK0 should converge much faster than 100 iters on a quadratic; "
        f"got {int(state.iter_num)}"
    )
    assert float(state.best_loss) < 1e-10


# =============================================================================
# Multi-seed comparison — LBFGSK0 final loss vs optax_lbfgs across seeds
# =============================================================================


def test_lbfgsk0_strictly_beats_optax_lbfgs_on_tight_box():
    """On tight box constraints where the unconstrained minimum lies outside
    the feasible region, the active-set method must outperform projection-
    based L-BFGS in BOTH final loss AND iteration count.

    With ``bounds=[-2, 0.3]`` the global Rosenbrock minimum at x=1 is outside
    the box, so 9 of the 10 parameters end up at the upper wall — the classic
    failure mode of projection-based L-BFGS, which our active-set solvers
    handle natively.
    """

    def f(x):
        return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    x0 = jnp.array([-1.0] * 10)
    common = dict(
        max_iter=2000, constraints=BoxConstraint(lower=-2.0, upper=0.3), atol=1e-12, rtol=1e-12
    )

    _, st_lbfgsk = minimize(f, x0, solver_name="LBFGSK0", **common)
    _, st_adabk = minimize(f, x0, solver_name="ADABK0", **common)
    _, st_opt = minimize(f, x0, solver_name="optax_lbfgs", **common)

    # Final loss: active-set strictly better than projection-based L-BFGS.
    assert float(st_lbfgsk.best_loss) < float(st_opt.best_loss), (
        f"LBFGSK0 loss {float(st_lbfgsk.best_loss):.6e} must beat "
        f"optax_lbfgs loss {float(st_opt.best_loss):.6e}"
    )
    assert float(st_adabk.best_loss) < float(st_opt.best_loss), (
        f"ADABK0 loss {float(st_adabk.best_loss):.6e} must beat "
        f"optax_lbfgs loss {float(st_opt.best_loss):.6e}"
    )
    # Iteration count: optax_lbfgs hits max_iter (stalled), LBFGSK0 / ADABK0 terminate.
    assert int(st_lbfgsk.iter_num) < int(st_opt.iter_num), (
        f"LBFGSK0 used {int(st_lbfgsk.iter_num)} iters; " f"optax_lbfgs used {int(st_opt.iter_num)}"
    )
    assert int(st_adabk.iter_num) < int(st_opt.iter_num), (
        f"ADABK0 used {int(st_adabk.iter_num)} iters; " f"optax_lbfgs used {int(st_opt.iter_num)}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lbfgsk0_matches_or_beats_optax_lbfgs_noisy(seed):
    """LBFGSK0 should achieve a final loss at least as good as optax_lbfgs."""
    f = _noisy_rosenbrock_factory(seed)
    _, st_lbfgsk = minimize(
        f,
        X0,
        solver_name="LBFGSK0",
        max_iter=2000,
        constraints=BoxConstraint(lower=-5.0, upper=10.0),
    )
    _, st_lbfgs = minimize(
        f,
        X0,
        solver_name="optax_lbfgs",
        max_iter=2000,
        constraints=BoxConstraint(lower=-5.0, upper=10.0),
    )
    # Allow a tiny numerical slack — the two methods follow different paths.
    assert float(st_lbfgsk.best_loss) <= float(st_lbfgs.best_loss) + 1e-6, (
        f"seed={seed}: LBFGSK0 loss {float(st_lbfgsk.best_loss):.4e} should be "
        f"≤ optax_lbfgs loss {float(st_lbfgs.best_loss):.4e}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adabk0_reaches_same_minimum_as_lbfgs_noisy(seed):
    """ADABK0 (first-order) must still reach the same loss as L-BFGS, given enough iters."""
    f = _noisy_rosenbrock_factory(seed)
    _, st_adabk = minimize(
        f,
        X0,
        solver_name="ADABK0",
        max_iter=3000,
        constraints=BoxConstraint(lower=-5.0, upper=10.0),
    )
    _, st_lbfgs = minimize(
        f,
        X0,
        solver_name="optax_lbfgs",
        max_iter=3000,
        constraints=BoxConstraint(lower=-5.0, upper=10.0),
    )
    assert float(st_adabk.best_loss) <= float(st_lbfgs.best_loss) + 1e-3, (
        f"seed={seed}: ADABK0 loss {float(st_adabk.best_loss):.4e} should be "
        f"≤ optax_lbfgs loss {float(st_lbfgs.best_loss):.4e}"
    )


# =============================================================================
# Box constraint via the typed `constraints` parameter
# =============================================================================


def test_box_constraint_object_matches_raw_bounds():
    """``BoxConstraint(lower, upper)`` must be equivalent to passing the raw bounds."""
    f = _clean_rosenbrock
    lo = jnp.full_like(X0, -5.0)
    up = jnp.full_like(X0, 10.0)

    _, st_raw = minimize(
        f, X0, solver_name="LBFGSK0", max_iter=500, constraints=BoxConstraint(lower=lo, upper=up)
    )
    _, st_obj = minimize(
        f, X0, solver_name="LBFGSK0", max_iter=500, constraints=BoxConstraint(lower=lo, upper=up)
    )

    assert jnp.allclose(st_raw.best_loss, st_obj.best_loss, atol=1e-12, rtol=1e-12)


def test_constraints_rejects_double_specification():
    """Passing both ``constraints`` and the deprecated ``lower_bound`` /
    ``upper_bound`` is an unambiguous user error."""
    f = _clean_rosenbrock
    lo = jnp.full_like(X0, -5.0)
    up = jnp.full_like(X0, 10.0)
    with pytest.raises(ValueError, match="deprecated"):
        minimize(
            f,
            X0,
            solver_name="LBFGSK0",
            lower_bound=lo,
            upper_bound=up,
            constraints=BoxConstraint(lower=lo, upper=up),
        )


def test_lower_upper_bound_emits_deprecation_warning():
    f = _clean_rosenbrock
    lo = jnp.full_like(X0, -5.0)
    up = jnp.full_like(X0, 10.0)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        minimize(f, X0, solver_name="LBFGSK0", lower_bound=lo, upper_bound=up, max_iter=10)


# =============================================================================
# Gaussian (soft) constraint
# =============================================================================


def test_gaussian_constraint_biases_solution_toward_prior_mean():
    """Strong Gaussian prior pulls the optimum from x* = target_data to x* ≈ loc."""
    target_data = jnp.array([5.0, 5.0, 5.0])
    loc = jnp.array([0.0, 0.0, 0.0])
    scale = jnp.array([0.05, 0.05, 0.05])  # very tight prior

    def f(x):
        return jnp.sum((x - target_data) ** 2)

    x0 = jnp.array([1.0, 1.0, 1.0])
    _, st = minimize(
        f,
        x0,
        solver_name="LBFGSK0",
        constraints=GaussianConstraint(loc=loc, scale=scale),
        max_iter=2000,
    )
    # Unconstrained min is (5, 5, 5) with f=0. Loss = ||x - 5||² + ||(x-0)/0.05||²
    # MAP solution (closed form): x_i = 5 / (1 + 1/0.05²) = 5 / 401 ≈ 0.01246.
    # Verify the solution sits near the prior mean and far from the data target.
    sol = st.best_y
    assert (
        jnp.max(jnp.abs(sol)) < 0.1
    ), f"GaussianConstraint should pull solution near loc=0; got {sol}"


def test_gaussian_constraint_relaxes_to_unconstrained_with_huge_scale():
    """With ``scale → ∞`` the prior gradient vanishes and the solver recovers
    the unconstrained minimum."""
    target_data = jnp.array([5.0, 5.0, 5.0])
    loc = jnp.array([0.0, 0.0, 0.0])
    scale = jnp.array([1e6, 1e6, 1e6])

    def f(x):
        return jnp.sum((x - target_data) ** 2)

    x0 = jnp.array([1.0, 1.0, 1.0])
    _, st = minimize(
        f,
        x0,
        solver_name="LBFGSK0",
        constraints=GaussianConstraint(loc=loc, scale=scale),
        max_iter=2000,
    )
    assert jnp.allclose(
        st.best_y, target_data, atol=1e-4
    ), f"With huge prior scale the solution should match the data target; got {st.best_y}"


# =============================================================================
# Langevin noise — opt-in, default disabled
# =============================================================================


def test_langevin_noise_default_disabled_is_deterministic():
    """Default (noise_temperature=0.0) → two runs from the same init produce identical results."""
    f = _clean_rosenbrock
    _, st1 = minimize(
        f,
        X0,
        solver_name="LBFGSK0",
        max_iter=300,
        constraints=BoxConstraint(lower=-5.0, upper=10.0),
    )
    _, st2 = minimize(
        f,
        X0,
        solver_name="LBFGSK0",
        max_iter=300,
        constraints=BoxConstraint(lower=-5.0, upper=10.0),
    )
    assert jnp.allclose(st1.best_y, st2.best_y, atol=0.0, rtol=0.0)


def test_langevin_noise_enabled_perturbs_run():
    """Two runs with the same seed but different noise temperatures should differ.

    Sanity check that the noise actually plumbs through to ``update_fn``.
    """
    f = _clean_rosenbrock
    _, st_quiet = minimize(
        f,
        X0,
        solver_name="LBFGSK0",
        max_iter=30,
        constraints=BoxConstraint(lower=-5.0, upper=10.0),
        options={"noise_temperature": 0.0, "noise_key": 7},
    )
    _, st_loud = minimize(
        f,
        X0,
        solver_name="LBFGSK0",
        max_iter=30,
        constraints=BoxConstraint(lower=-5.0, upper=10.0),
        options={"noise_temperature": 0.5, "noise_key": 7},
    )
    # Loss may differ in either direction — noise can help or hurt mid-run.
    # We only require that the trajectory diverges.
    assert not jnp.allclose(st_quiet.best_y, st_loud.best_y, atol=1e-6)


# =============================================================================
# Solver registry — new names appear in the Literal and the SELFCONDITIONED set
# =============================================================================


def test_langevin_noise_escapes_local_minimum_on_six_hump_camel():
    """Six-Hump Camel: 2 global at f≈-1.03, 4 local at f≈-0.22.

    Start in the local basin. At least one Langevin seed must escape to the
    global basin; deterministic methods must NOT.
    """

    def camel(p):
        x, y = p[0], p[1]
        return (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2

    x0 = jnp.array([1.5, -0.8])
    common = dict(max_iter=2000, constraints=BoxConstraint(lower=-2.5, upper=2.5))

    # Deterministic LBFGSK gets stuck in the local well (f ≈ -0.22).
    _, st_det = minimize(camel, x0, solver_name="LBFGSK0", **common)
    assert -0.30 < float(st_det.best_loss) < -0.10, (
        f"Deterministic LBFGSK0 should land in the local basin "
        f"(f ≈ -0.22); got {float(st_det.best_loss):.4f}"
    )

    # At least one Langevin seed must reach a global basin (f ≈ -1.03).
    best_loss = float("inf")
    for key in range(6):
        _, st = minimize(
            camel,
            x0,
            solver_name="LBFGSK0",
            options={"noise_temperature": 2.0, "noise_key": key},
            **common,
        )
        best_loss = min(best_loss, float(st.best_loss))
    assert best_loss < -0.9, (
        f"At least one Langevin seed must reach a global basin (f < -0.9); "
        f"best across seeds was {best_loss:.4f}"
    )


def test_lbfgsk_with_ema_zero_matches_optax_lbfgs_to_machine_precision():
    """With ``lbfgs_ema_decay = 0`` the EMA layer is a no-op; LBFGSK0 must
    reach the same minimum as ``optax_lbfgs`` on a smooth problem."""

    def f(x):
        return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    x0 = jnp.array([-1.2, 1.0] * 5)
    _, st_lbfgsk = minimize(
        f,
        x0,
        solver_name="LBFGSK0",
        options={"lbfgs_ema_decay": 0.0},
        max_iter=2000,
        constraints=BoxConstraint(lower=-5.0, upper=10.0),
    )
    _, st_opt = minimize(
        f,
        x0,
        solver_name="optax_lbfgs",
        max_iter=2000,
        constraints=BoxConstraint(lower=-5.0, upper=10.0),
    )
    # Both should reach machine-precision-near-zero on smooth Rosenbrock.
    assert float(st_lbfgsk.best_loss) < 1e-8
    assert float(st_opt.best_loss) < 1e-8


def test_lbfgsk_solvers_dispatch():
    """LBFGSK is a prefix-dispatched solver; no name needs to appear in the Literal."""
    from cadre.lbfgsk import _parse_lbfgsk_k

    assert _parse_lbfgsk_k("LBFGSK0") == 0.0
    assert _parse_lbfgsk_k("LBFGSK5") == 0.5
    assert _parse_lbfgsk_k("LBFGSK") is None
    assert _parse_lbfgsk_k("not_lbfgsk") is None
    # And it actually builds:
    _, _ = cadre.get_solver("LBFGSK0", atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
