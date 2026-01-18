import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Sequence, Optional
import math

# HELPER FUNCTIONS #

def atanh(x, eps=1e-6):
    x = jnp.clip(x, -1 + eps, 1 - eps)
    return 0.5 * (jnp.log1p(x) - jnp.log1p(-x))

# MLP POLICIES #

class TanhGaussianPolicy(nn.Module):
    state_dim: int
    action_dim: int
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "relu"

    def setup(self):
        activation_map = {
            "relu": nn.relu,
            "leakyrelu": nn.leaky_relu,
            "gelu": nn.gelu,
            "selu": nn.selu,
            "silu": nn.silu,
            "mish": lambda x: x * jnp.tanh(nn.softplus(x)),
            "tanh": nn.tanh,
            "sigmoid": nn.sigmoid,
        }
        self.act_fn = activation_map.get(self.activation, nn.relu)
        
        self.latent_layers = [nn.Dense(h) for h in self.hidden_sizes]
        self.mu_layer = nn.Dense(self.action_dim)
        self.log_std_layer = nn.Dense(self.action_dim)

    def __call__(self, state):
        x = state
        for layer in self.latent_layers:
            x = layer(x)
            x = self.act_fn(x)
        
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = jnp.clip(log_std, -20, 2)
        return mu, log_std

    def sample(self, state, rng):
        mu, log_std = self.__call__(state)
        std = jnp.exp(log_std)
        
        # Reparameterization trick
        eps = jax.random.normal(rng, shape=mu.shape)
        action_pre = mu + std * eps
        
        action = jnp.tanh(action_pre)
        
        # Log prob calculation
        log_prob_gauss = -0.5 * (((action_pre - mu) / (std + 1e-6)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
        log_prob_gauss = jnp.sum(log_prob_gauss, axis=-1)
        
        log_det_jac = jnp.sum(jnp.log(1 - action**2 + 1e-6), axis=-1)
        log_prob = log_prob_gauss - log_det_jac
        
        return action, log_prob
    
    def sample_without_probs(self, state, rng):
        mu, log_std = self.__call__(state)
        std = jnp.exp(log_std)
        
        # Reparameterization trick
        eps = jax.random.normal(rng, shape=mu.shape)
        action_pre = mu + std * eps
        
        action = jnp.tanh(action_pre)
        
        return action

    def sample_det(self, state):
        mu, log_std = self.__call__(state)
        action_pre = mu
        action = jnp.tanh(action_pre)
    
        return action

    def log_probs(self, state, action):
        action_pre = atanh(action)
        mu, log_std = self.__call__(state)
        std = jnp.exp(log_std)
        
        log_prob_gauss = -0.5 * (((action_pre - mu) / (std + 1e-6)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
        log_prob_gauss = jnp.sum(log_prob_gauss, axis=-1)
        
        # The PyTorch code used a stable softplus formula here, replicating it:
        # 2 * (log(2) - x - softplus(-2x))
        log_det_jac = jnp.sum(2 * (jnp.log(2) - action_pre - nn.softplus(-2 * action_pre)), axis=-1)
        
        log_prob = log_prob_gauss - log_det_jac
        return log_prob

class DeterministicPolicy(nn.Module):
    state_dim: int
    action_dim: int
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "relu"

    @nn.compact
    def __call__(self, state):
        activation_map = {
            "relu": nn.relu,
            "leakyrelu": nn.leaky_relu,
            "gelu": nn.gelu,
            "selu": nn.selu,
            "silu": nn.silu,
            "mish": lambda x: x * jnp.tanh(nn.softplus(x)),
            "tanh": nn.tanh,
            "sigmoid": nn.sigmoid,
        }
        act_fn = activation_map.get(self.activation, nn.relu)

        x = state
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            x = act_fn(x)
            
        x = nn.Dense(self.action_dim)(x)
        x = jnp.tanh(x)
        return x
