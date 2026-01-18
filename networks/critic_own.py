import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Callable, Optional

class Q_network(nn.Module):
    state_dim: int
    action_dim: int
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "relu"
    layer_norm: bool = False

    @nn.compact
    def __call__(self, state, action):
        # Activation map
        activation_map = {
            "relu": nn.relu,
            "leakyrelu": nn.leaky_relu,
            "gelu": nn.gelu,
            "selu": nn.selu,
            "silu": nn.silu,
            "tanh": nn.tanh,
            "sigmoid": nn.sigmoid,
        }
        
        # Handle mish if not in nn (older flax versions)
        act_fn = activation_map.get(self.activation, nn.relu)
        x = jnp.concatenate([state, action], axis=-1)

        for hidden_size in self.hidden_sizes:
            x = nn.Dense(features=hidden_size)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = act_fn(x)

        x = nn.Dense(features=1)(x)
        return x.squeeze(-1) # Match pytorch shape (batch,)
    
class CombinedCritics(nn.Module):
    state_dim: int
    action_dim: int
    n_critics: int
    critic_kwargs: dict

    @nn.compact
    def __call__(self, state, action):
        # We use vmap to create n_critics independent networks
        # variable_axes={'params': 0} indicates the 0th axis of parameters corresponds to the critic index
        # split_rngs={'params': True} ensures each critic gets different initialization
        VectorizedCritic = nn.vmap(
            Q_network,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics
        )
        
        critics = VectorizedCritic(
            state_dim=self.state_dim, 
            action_dim=self.action_dim, 
            **self.critic_kwargs
        )
        
        # Returns shape: (n_critics, batch)
        return critics(state, action)