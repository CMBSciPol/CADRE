# CADRE

**Constraint-Aware Descent Routine Executor** — JAX-native constrained optimization.

[![PyPI](https://img.shields.io/pypi/v/jax-cadre)](https://pypi.org/project/jax-cadre/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs: minimization](https://img.shields.io/badge/docs-minimization-blue)](https://furax-cs.readthedocs.io/en/latest/minimization.html)

CADRE provides a unified interface to multiple JAX-compatible optimization backends, with first-class support for box-constrained problems via AdaTopK, an active-set method (ADABK family).

This is the minimizer used in [Furax-CS](https://github/CMBSciPol/furax-cs) package for CMB component separation.

## Installation

```bash
pip install jax-cadre
```

With optional scipy solvers (`scipy_tnc`, `scipy_cobyqa`):

```bash
pip install jax-cadre[scipy]
```

## Quick start

```python
from cadre import minimize
import jax.numpy as jnp

def loss(params, target):
    return jnp.sum((params - target) ** 2)

target = jnp.array([1.0, 2.0, 3.0])
lower  = jnp.zeros(3)
upper  = jnp.ones(3) * 5.0

params, state = minimize(
    loss,
    init_params=jnp.zeros(3),
    solver_name="ADABK0",   # or "optax_lbfgs"
    lower_bound=lower,
    upper_bound=upper,
    target=target,
)

print(f"Optimal params: {params}")

```

## Solvers

| Solver | Description |
|--------|-------------|
| `ADABK0` | AdaTopK: active set + AdaBelief, 1 constraint released/step. **Best for noisy landscapes.** |
| `ADABK{N}` | AdaTopK, up to `N×10 %` constraints released/step. |
| `optax_lbfgs` | L-BFGS with zoom linesearch. **Best for smooth landscapes.** |
| `adam`, `adabelief`, `adaw`, `sgd` | First-order optax solvers with optional projection. |
| `optimistix_bfgs/lbfgs/ncg_*` | Optimistix solvers. |
| `scipy_tnc`, `scipy_cobyqa` | Scipy solvers via jaxopt *(requires `cadre[scipy]`)*. |

Full solver documentation and ADABK internals:
**[docs](https://furax-cs.readthedocs.io/en/latest/minimization.html)**

## Advanced usage

`AdaTopK` is an `optimistix` minimiser and can be used directly:

```python
import optimistix as optx
from cadre import AdaTopK

solver = AdaTopK(rtol=0.0, atol=1e-10, lower=lower, upper=upper)
sol = optx.minimise(lambda p, t: loss(p, t), solver, jnp.zeros(3), target, max_steps=1000, throw=False)
print(sol.value, sol.stats["num_steps"])
```

## License

MIT — see [LICENSE](LICENSE).
