import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import flax.linen as nn
import optax
import numpy as np
import random
from time import time
from tqdm import tqdm
from collections import deque, namedtuple
import mlflow

from common.replaybuffer import ReplayBuffer
from common.logger import MLFlowLogger
from common.utils import load_demo_trajectories, load_demo_trajectories_parallel

from networks.critic_own import Q_network, CombinedCritics
from networks.actor_own import DeterministicPolicy, TanhGaussianPolicy
from flax.training.train_state import TrainState

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'termination', 'truncation', 'next_state'])

class AlphaModule(nn.Module):
    target_entropy: float
    
    @nn.compact
    def __call__(self):
        log_alpha = self.param('log_alpha', lambda key: jnp.zeros((1,)))
        return log_alpha
    
def interleave_dicts(a: dict, b: dict) -> dict:
    """
    Combines two dictionaries by interleaving arrays along axis 0.
    Returns Numpy arrays (JAX will convert them on device transfer).
    """
    combined = {}
    for k, va in a.items():
        vb = b[k]
        if isinstance(va, dict):
            combined[k] = interleave_dicts(va, vb)
            continue
            
        # Efficient Numpy Interleaving
        len_a, len_b = va.shape[0], vb.shape[0]
        out_shape = (len_a + len_b, *va.shape[1:])
        out = np.empty(out_shape, dtype=va.dtype)
        
        if len_a == len_b:
            out[0::2] = va
            out[1::2] = vb
        else:
            out = np.concatenate([va, vb], axis=0)
            
        combined[k] = out # Return numpy array
    return combined



class SACPDAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        tau=0.005,
        gamma=0.99,
        alpha=0.2,
        lr=3e-4,
        batch_size=256,
        buffer_size=1_000_000,
        n_steps=1,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        policy_delay_update=1,
        target_entropy=None,
        seed=0,
        stats_window_size=100,
        n_envs=4,
        per_batch_demo=1.0,
        n_critics=5,
        m_critics=2,
        logger_name="mlflow",
        policy_kwargs=dict(),
        critic_kwargs=dict(layer_norm=True),
        experiment_name="",
        run_name="",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.n_steps = n_steps
        assert n_steps == 1, "Support for n_step != 1 is not available!!!"
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.policy_delay_update = policy_delay_update
        self.per_batch_demo = per_batch_demo
        self.m_critics = m_critics
        self.n_critics = n_critics
        self.logger_name = logger_name
        self.policy_kwargs = policy_kwargs
        self.critic_kwargs = critic_kwargs
        
        # Logging
        self._count_total_gradients_taken = 0
        self._ep_rewards = deque(maxlen=stats_window_size)
        self._ep_lengths = deque(maxlen=stats_window_size)
        self._start_time = time()

        # Random Keys
        self.rng = jax.random.PRNGKey(seed if seed is not None else 0)
        np.random.seed(seed)
        random.seed(seed)

        # Initialize Networks & Optimizers
        self.rng, actor_key, critic_key, alpha_key = jax.random.split(self.rng, 4)
        
        # 1. Actor
        self.actor_def = TanhGaussianPolicy(state_dim=state_dim, action_dim=action_dim, **policy_kwargs)
        dummy_obs = jnp.zeros((1, state_dim))
        actor_params = self.actor_def.init(actor_key, dummy_obs)
        self.actor_state = TrainState.create(
            apply_fn=self.actor_def.apply,
            params=actor_params,
            tx=optax.adam(lr)
        )

        # 2. Critics
        self.critics_def = CombinedCritics(
            state_dim=state_dim, action_dim=action_dim, 
            n_critics=n_critics, critic_kwargs=critic_kwargs
        )
        dummy_action = jnp.zeros((1, action_dim))
        critic_params = self.critics_def.init(critic_key, dummy_obs, dummy_action)
        self.critics_state = TrainState.create(
            apply_fn=self.critics_def.apply,
            params=critic_params,
            tx=optax.adam(lr)
        )
        
        # Target Critics (Just params, no optimizer)
        self.target_critics_params = critic_params

        # 3. Alpha (Temperature)
        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = target_entropy
            
        self.alpha_def = AlphaModule(target_entropy=self.target_entropy)
        alpha_params = self.alpha_def.init(alpha_key)
        self.alpha_state = TrainState.create(
            apply_fn=self.alpha_def.apply,
            params=alpha_params,
            tx=optax.adam(lr)
        )

        # Replay Buffers
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size, int(batch_size*self.gradient_steps), gamma, n_envs)
        self.replay_buffer_demo = ReplayBuffer(state_dim, action_dim, buffer_size, int(batch_size*self.gradient_steps*self.per_batch_demo), gamma, 1)

        # Logger
        if logger_name == "mlflow":
            self.logger = MLFlowLogger(uri="http://127.0.0.1:5000", experiment_name=experiment_name, run_name=run_name)
            
        # Hparams for logging
        self.hparams = {
            "state_dim": self.state_dim, "action_dim": self.action_dim, "tau": self.tau,
            "gamma": self.gamma, "lr": self.lr, "batch_size": self.batch_size,
            "n_critics": n_critics, "m_critics": m_critics, "seed": seed
        }


    @staticmethod
    @partial(jax.jit, static_argnames="apply_fn")
    def _sample_action(apply_fn, params, obs, key):
        actions = apply_fn(params, obs, key, method=TanhGaussianPolicy.sample_without_probs)
        return actions
    
    @staticmethod
    @partial(jax.jit, static_argnames="apply_fn")
    def _sample_action_det(apply_fn, params, obs):
        actions = apply_fn(params, obs, method=TanhGaussianPolicy.sample_det)
        return actions

    def select_action(self, state, deterministic=True):
        state = jnp.array(state)
        if state.ndim == 1: state = state[None, ...]
        
        if deterministic:
            action = SACPDAgent._sample_action_det(self.actor_state.apply_fn, self.actor_state.params, state)
        else:
            self.rng, key = jax.random.split(self.rng)
            action = SACPDAgent._sample_action(self.actor_state.apply_fn, self.actor_state.params, state, key)
            
        return np.array(action)
    
    def load_demo_trajectories(self, demo_file, demo_env, nds=40, nds_name="CGL", n_load=-1):
        load_demo_trajectories_parallel([self.replay_buffer_demo], demo_file, demo_env, nds, nds_name, self.gamma, n_load=n_load)
        return self.replay_buffer_demo
    
    #  Single Step Updates #
    @staticmethod
    # @partial(jax.jit, static_argnames=("n_critics", "m_critics"))
    def _update_critic_step(
        critics_state, target_critics_params, actor_state, alpha_state,
        batch, rng,
        gamma, n_critics, m_critics
    ):
      
        # --- 1. Compute Targets ---
        rng, key_target_policy, key_subset = jax.random.split(rng, 3)
        
        # Sample next actions from current actor (to compute next log probs)
        # Note: We use the current actor state for the target policy part as per standard SAC
        next_actions, next_log_probs = actor_state.apply_fn(
            actor_state.params, batch["next_observations"], key_target_policy, method=TanhGaussianPolicy.sample
        )
        
        # Get target Q-values using Target Critic Params
        target_qs_all = critics_state.apply_fn(target_critics_params, batch["next_observations"], next_actions) # (N, B)
        
        # Select m random critics for robust estimation
        idx = jax.random.choice(key_subset, n_critics, shape=(m_critics,), replace=False)
        target_qs_subset = target_qs_all[idx] # (M, B)
        
        # Min Q calculation (Clipped Double Q-Learning)
        min_target_q = jnp.min(target_qs_subset, axis=0)
        
        # Retrieve Alpha
        log_alpha = alpha_state.apply_fn(alpha_state.params)
        alpha = jnp.exp(log_alpha)
        
        # Compute Bellman Target
        # target = r + gamma * (1 - d) * (min_Q - alpha * log_pi)
        q_backup = batch["rewards"] + (1 - batch["terminations"]) * (gamma) * (min_target_q - alpha * next_log_probs)
        
        # --- 2. Calculate Critic Loss & Update ---
        def critic_loss_fn(p):
            # Calculate Q-values for current observations and actions
            current_qs = critics_state.apply_fn(p, batch["observations"], batch["actions"]) # (N, B)
            
            # Loss is sum of MSEs over all N critics
            # We broaden q_backup to (1, B) to broadcast against (N, B)
            loss = jnp.mean(jnp.square(current_qs - q_backup[None, :]))
            return loss

        critic_grad_fn = jax.value_and_grad(critic_loss_fn)
        critic_loss, critic_grads = critic_grad_fn(critics_state.params)
        new_critics_state = critics_state.apply_gradients(grads=critic_grads)
        
        return new_critics_state, critic_loss
    
    @staticmethod
    # @jax.jit
    def _update_actor_and_alpha_step(
        actor_state, critics_state, alpha_state,
        batch, rng,
        target_entropy
    ):
        states = batch["observations"]
        rng, key_actor = jax.random.split(rng)
        
        # Retrieve Alpha
        log_alpha = alpha_state.apply_fn(alpha_state.params)
        alpha = jnp.exp(log_alpha)

        # --- 1. Update Actor ---
        def actor_loss_fn(p):
            # Sample actions from the policy
            curr_actions, curr_log_probs = actor_state.apply_fn(p, states, key_actor, method=TanhGaussianPolicy.sample)
            
            # Get Q-values from the *updated* critic
            # We average the Q-values from all critics to reduce variance
            qs_pi = critics_state.apply_fn(critics_state.params, states, curr_actions)
            q_pi = jnp.mean(qs_pi, axis=0)
            
            # Actor loss: alpha * log_pi - Q
            loss = jnp.mean(jax.lax.stop_gradient(alpha) * curr_log_probs - q_pi)
            return loss, curr_log_probs

        # Compute gradients (has_aux=True to keep log_probs for alpha update)
        (actor_loss, log_probs_for_alpha), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        
        # --- 2. Update Alpha ---
        def alpha_loss_fn(p):
            cur_log_alpha = alpha_state.apply_fn(p)
            # Alpha loss: -log_alpha * (log_pi + target_entropy)
            # We use stop_gradient on log_probs because alpha update shouldn't affect actor
            return -jnp.mean(cur_log_alpha * (jax.lax.stop_gradient(log_probs_for_alpha) + target_entropy))
            
        alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(alpha_state.params)
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        
        return new_actor_state, new_alpha_state, actor_loss, alpha_loss, alpha
    
    # Fused Update
    @staticmethod
    @partial(jax.jit, static_argnames=("gradient_steps", "n_critics", "m_critics", "policy_delay_update"))
    def batch_update(
        actor_state, critics_state, target_critics_params, alpha_state,
        batch, rng,
        gamma, tau, target_entropy,
        gradient_steps, n_critics, m_critics, policy_delay_update
    ):
        """
        Unrolled update loop for maximum speed.
        Assumes batch is shape (gradient_steps * batch_size, ...).
        """
        
        # Helper to slice the big batch into micro-batches
        def get_slice(x, i):
            step_size = x.shape[0] // gradient_steps
            return jax.lax.dynamic_slice_in_dim(x, i * step_size, step_size, axis=0)

        # We manually unroll the loop using Python `for`
        # This creates a larger computation graph but allows XLA to optimize across steps
        # and avoids the overhead of lax.scan for small N (N < 50).
        
        # We need to track the LAST metrics for logging
        last_metrics = {} 

        for i in range(gradient_steps):
            rng, r_critic, r_actor = jax.random.split(rng, 3)
            
            # 1. Slice Batch
            step_batch = jax.tree_util.tree_map(lambda x: get_slice(x, i), batch)
            
            # 2. Update Critic
            critics_state, c_loss = SACPDAgent._update_critic_step(
                critics_state, target_critics_params, actor_state, alpha_state, step_batch, r_critic, gamma, n_critics, m_critics
            )
                
            # Soft Update Targets
            target_critics_params = optax.incremental_update(critics_state.params, target_critics_params, tau)
            
        actor_state, alpha_state, a_loss, al_loss, curr_alpha = SACPDAgent._update_actor_and_alpha_step(
            actor_state, critics_state, alpha_state, step_batch, r_actor, target_entropy
        )

        last_metrics = {
            "critic_loss": c_loss, "actor_loss": a_loss, "alpha_loss": al_loss, "alpha": curr_alpha
        }

        return actor_state, critics_state, target_critics_params, alpha_state, last_metrics
    
    def train(self, env, total_training_steps=1_000_000, learning_starts=2_000, progress_bar=True, verbose=1, log_interval=5, log_interval_metrics=5000, callback=None):
        self.total_training_steps = total_training_steps
        self.learning_starts = learning_starts
        
        if callback: callback.on_training_start(self)
        if self.gradient_steps == -1: self.gradient_steps = env.num_envs
        
        pbar = tqdm(total=total_training_steps) if progress_bar else None
        if hasattr(self, "logger"):
            self.logger.start()
            self.logger.log_params(self.hparams)

        obs, _ = env.reset()
        _episode_start = np.zeros(env.num_envs, dtype=bool)
        _episode_rewards = np.zeros(env.num_envs)
        _episode_lengths = np.zeros(env.num_envs)
        
        self._total_timesteps_ran = 0
        self.logger_count = 1

        while self._total_timesteps_ran <= total_training_steps:
            # Action Selection
            self.rng, key = jax.random.split(self.rng)
            actions = self.select_action(obs, deterministic=False)

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            dones = np.logical_or(terminations, truncations)

            for i in range(env.num_envs):
                if not _episode_start[i]:
                    self.replay_buffer.add(obs[i], actions[i], rewards[i], terminations[i], truncations[i], next_obs[i])
                else:
                    self.logger_count += 1
                    self._ep_lengths.append(_episode_lengths[i])
                    self._ep_rewards.append(_episode_rewards[i])
                    _episode_rewards[i], _episode_lengths[i] = 0, -1

            _episode_rewards += rewards
            _episode_lengths += 1
            self._total_timesteps_ran += env.num_envs
            obs = next_obs
            _episode_start = dones

            # Update Step
            if self._total_timesteps_ran >= learning_starts and self._total_timesteps_ran % self.train_freq == 0:
                # Sample Buffers
                batch = self.replay_buffer.sample()
                batch_demo = self.replay_buffer_demo.sample()

                # Combine & Cast to JAX Arrays
                combined_batch = interleave_dicts(batch, batch_demo)

                # Update
                self.rng, update_key = jax.random.split(self.rng)
                
                self.rng, key = jax.random.split(self.rng)
                (self.actor_state, self.critics_state, self.target_critics_params, self.alpha_state, metrics) = \
                SACPDAgent.batch_update(
                    self.actor_state, self.critics_state, self.target_critics_params, self.alpha_state,
                    combined_batch, update_key,
                    self.gamma, self.tau, self.target_entropy,
                    self.gradient_steps, self.n_critics, self.m_critics, self.policy_delay_update
                )

                self._count_total_gradients_taken += self.gradient_steps                    

            # Logging
            if self._total_timesteps_ran >= learning_starts and self.logger_count % log_interval == 0:
                mean_rew = np.mean(self._ep_rewards) if self._ep_rewards else 0
                mean_len = np.mean(self._ep_lengths) if self._ep_lengths else 0
                fps = self._total_timesteps_ran / (time() - self._start_time)

                self.logger.log_metric("rollout/mean_episode_length", mean_len, step=self._total_timesteps_ran)
                self.logger.log_metric("rollout/mean_episode_reward", mean_rew, step=self._total_timesteps_ran)
                self.logger.log_metric("rollout/frames_per_second", fps, step=self._total_timesteps_ran)
                
                if verbose:
                    tqdm.write("-"*50)
                    tqdm.write(f" Step: {self._total_timesteps_ran:<8d}")
                    tqdm.write(f" MeanEpLen: {mean_len:.2f}")
                    tqdm.write(f" MeanEpRew: {mean_rew:.2f}")
                    tqdm.write("-"*50)
                
                self.logger_count = 1

            if self._total_timesteps_ran >= learning_starts and self._count_total_gradients_taken % log_interval_metrics == 0:
                self.logger.log_metric("training/actor_loss", metrics['actor_loss'].item(), step=self._total_timesteps_ran)
                self.logger.log_metric("training/critic_loss", metrics['critic_loss'].item(), step=self._total_timesteps_ran)
                self.logger.log_metric("training/alpha_loss", metrics['alpha_loss'].item(), step=self._total_timesteps_ran)
                self.logger.log_metric("training/alpha", metrics['alpha'].item(), step=self._total_timesteps_ran)

                if verbose:
                    tqdm.write("-"*50)
                    tqdm.write(f" Actor_Loss: {metrics['actor_loss'].item():<2f}")
                    tqdm.write(f" Critic_Loss: {metrics['critic_loss'].item():.2f}")
                    tqdm.write(f" Alpha_Loss: {metrics['alpha_loss'].item():.2f}")
                    tqdm.write(f" Alpha: {metrics['alpha'].item():.2f}")
                    tqdm.write("-"*50)

            if callback: callback.on_step(self._total_timesteps_ran,self)
            if pbar: pbar.update(env.num_envs)

        if callback: callback.on_training_end(self)
        return self.actor_state 
