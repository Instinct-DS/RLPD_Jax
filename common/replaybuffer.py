# Replay buffer to store transitions and support sampling #

import numpy as np
from collections import deque, namedtuple
import random

# Replay buffer with n-step support
Transition = namedtuple('Transition', ['observations', 'actions', 'rewards', 'terminations', 'truncations', 'next_observations'])

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size=1_000_000, batch_size=256, gamma=0.99, n_envs=4):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.gamma = gamma
        self.ptr, self.size = 0, 0

        self.observations = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.terminations = np.zeros((buffer_size,), dtype=np.float32)
        self.truncations = np.zeros((buffer_size,), dtype=np.float32)
        self.masks = np.zeros((buffer_size,), dtype=np.float32)

    def add(self, observations, actions, rewards, terminations, truncations, next_observations):
        # Store to main buffer
        self.observations[self.ptr] = observations
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_observations[self.ptr] = next_observations
        self.terminations[self.ptr] = terminations
        self.truncations[self.ptr] = truncations

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, size = None):
        if size is None:
            size = self.size
        # idx = np.random.randint(0, size, size=self.batch_size)
        idx = np.random.permutation(size)[:self.batch_size]
        batch = dict(
            observations = self.observations[idx],
            actions = self.actions[idx],
            rewards = self.rewards[idx],
            next_observations = self.next_observations[idx],
            terminations = self.terminations[idx],
            truncations = self.truncations[idx],
            masks = 1.0 - self.terminations[idx]
        )
        return batch
    
    def sample_batches(self, num_grads=10):
        num_grads = min(self.size//self.batch_size, num_grads)
        sample_size = num_grads * self.batch_size

        indices = np.random.randint(0, self.size, size=sample_size)
        np.random.shuffle(indices)

        def _generator():
            for i in range(num_grads):
                idx = indices[i*self.batch_size:(i+1)*self.batch_size]
                batch = dict(
                    observations = self.observations[idx],
                    actions = self.actions[idx],
                    rewards = self.rewards[idx],
                    next_observations = self.next_observations[idx],
                    terminations = self.terminations[idx],
                    truncations = self.truncations[idx],
                    masks = 1.0 - self.terminations[idx]
                )

                yield batch

    def current_size(self):
        return self.size