from typing import Any, Optional

import distrax
import jax
import jax.numpy as jnp

# Inspired by
# https://github.com/deepmind/acme/blob/300c780ffeb88661a41540b99d3e25714e2efd20/acme/jax/networks/distributional.py#L163
# but modified to only compute a mode.


class TanhTransformedDistribution(distrax.Transformed):
    def __init__(self, distribution: distrax.Distribution, validate_args: bool = False):
        bijector = distrax.Block(distrax.Tanh(), ndims=1)
        super().__init__(
            distribution=distribution, bijector=bijector,
        )

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties