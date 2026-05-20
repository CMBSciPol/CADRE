"""Constraint types for the CADRE optimization framework.

Provides box (uniform) and Gaussian (soft) constraint representations that
are compatible with JAX pytrees and equinox modules.

References:
    Kabalan et al. (2025), arXiv:2604.08463 — AdaTopK active-set method
"""

from __future__ import annotations

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import Array, Float, PyTree


class BoxConstraint(eqx.Module):
    """Box (uniform) constraints: x ∈ [lower, upper] for each parameter.

    Equivalent to passing lower_bound and upper_bound to minimize() directly.
    lower and upper must have the same pytree structure as the parameters.
    """

    lower: PyTree[Float[Array, " P"]]
    upper: PyTree[Float[Array, " P"]]


class GaussianConstraint(eqx.Module):
    """Soft Gaussian prior: adds (x − loc) / scale² to the objective gradient.

    Encourages x ≈ loc with uncertainty controlled by scale. No hard bounds are
    imposed — the active-set framework runs but wall constraints are never
    triggered. Acts as ridge regularization toward the prior mean.

    loc and scale must have the same pytree structure as the parameters.

    References:
        Standard Gaussian MAP / ridge regularization.
    """

    loc: PyTree[Float[Array, " P"]]
    scale: PyTree[Float[Array, " P"]]


Constraint = BoxConstraint | GaussianConstraint


def validate_constraint(c: Constraint) -> None:
    """Validate that constraint trees have consistent pytree structure.

    Raises:
        ValueError: If lower/upper or loc/scale have mismatched pytree structures.
    """
    if isinstance(c, BoxConstraint):
        try:
            jtu.tree_map(lambda _a, _b: None, c.lower, c.upper)
        except ValueError as e:
            raise ValueError(
                f"BoxConstraint: lower and upper must have the same pytree structure. {e}"
            ) from e
    elif isinstance(c, GaussianConstraint):
        try:
            jtu.tree_map(lambda _a, _b: None, c.loc, c.scale)
        except ValueError as e:
            raise ValueError(
                f"GaussianConstraint: loc and scale must have the same pytree structure. {e}"
            ) from e
