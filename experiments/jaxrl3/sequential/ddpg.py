import functools
from typing import Any, Dict, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl3.agents.agent import Agent


class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


@functools.partial(jax.jit, static_argnames=("gamma"))
def update_critic(
    actor_state: TrainState,
    qf1_state: TrainState,
    observations: np.ndarray,
    actions: np.ndarray,
    next_observations: np.ndarray,
    rewards: np.ndarray,
    terminations: np.ndarray,
    gamma: float,

):
    next_state_actions = (actor_state.apply_fn({"params": actor_state.target_params}, next_observations)).clip(-1, 1)  # TODO: proper clip
    qf1_next_target = qf1_state.apply_fn({"params": qf1_state.target_params}, next_observations, next_state_actions).reshape(-1)
    next_q_value = (rewards + (1 - terminations) * gamma * (qf1_next_target)).reshape(-1)

    def mse_loss(params):
        qf1_a_values = qf1_state.apply_fn({"params": params}, observations, actions).squeeze()
        return ((qf1_a_values - next_q_value) ** 2).mean(), qf1_a_values.mean()

    (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(mse_loss, has_aux=True)(qf1_state.params)
    qf1_state = qf1_state.apply_gradients(grads=grads1)

    return qf1_state, qf1_loss_value, qf1_a_values


@functools.partial(jax.jit, static_argnames=("tau"))
def update_actor(
    actor_state: TrainState,
    qf1_state: TrainState,
    observations: np.ndarray,
    tau: float
):
    def actor_loss(params):
        return -qf1_state.apply_fn({"params": qf1_state.params}, observations, actor_state.apply_fn({"params": params}, observations)).mean()

    actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=grads)
    actor_state = actor_state.replace(
        target_params=optax.incremental_update(actor_state.params, actor_state.target_params, tau)
    )

    qf1_state = qf1_state.replace(
        target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, tau)
    )
    return actor_state, qf1_state, actor_loss_value


@jax.jit
def eval_actions_jit(
    actor_state: TrainState,
    observations: np.ndarray,
) -> jnp.ndarray:
    actions = actor_state.apply_fn({"params": actor_state.params}, observations)
    return actions


@jax.jit
def sample_actions_jit(
    rng: Any,
    actor_state: TrainState,
    observations: np.ndarray,
    exploration_noise: jnp.ndarray,
    action_low_high: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[Any, jnp.ndarray]:
    rng, noise_key = jax.random.split(rng)
    actions = actor_state.apply_fn({"params": actor_state.params}, observations)
    noise = 0 + exploration_noise * jax.random.normal(noise_key)
    actions = (actions + noise).clip(*action_low_high)
    return rng, actions


class DDPGLearner(Agent):
    def __init__(
        self,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        policy_frequency: int = 2,
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: float = 0.1,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        self.tau = tau
        self.gamma = gamma

        observations = observation_space.sample()
        actions = action_space.sample()
        action_scale = jnp.array((action_space.high - action_space.low) / 2.0)
        action_bias = jnp.array((action_space.high + action_space.low) / 2.0)
        action_dim = action_space.shape[-1]

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, qf1_key, temp_key = jax.random.split(rng, 4)

        actor = Actor(
            action_dim=action_dim,  # np.prod(action_space.shape),
            action_scale=action_scale,
            action_bias=action_bias,
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, observations)["params"],
            target_params=actor.init(actor_key, observations)["params"],
            tx=optax.adam(learning_rate=actor_lr),
        )
        qf1 = QNetwork()
        qf1_state = TrainState.create(
            apply_fn=qf1.apply,
            params=qf1.init(qf1_key, observations, actions)["params"],
            target_params=qf1.init(qf1_key, observations, actions)["params"],
            tx=optax.adam(learning_rate=critic_lr),
        )
        self._actor = actor_state
        self._qf1 = qf1_state
        self._rng = rng
        self._update_step = 0
        self._policy_frequency = policy_frequency
        self._exploration_noise = action_scale * exploration_noise
        self._action_low_high = (jnp.array(action_space.low), jnp.array(action_space.high))
        self._update_info = {"actor_loss": None, "critic_loss": None}

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        qf1_state, qf1_loss_value, qf1_a_values = update_critic(
            self._actor,
            self._qf1,
            batch["observations"],
            batch["actions"],
            batch["next_observations"],
            batch["rewards"],
            batch["dones"],
            self.gamma,
        )
        self._qf1 = qf1_state
        self._update_info["critic_loss"] = qf1_loss_value
        if self._update_step % self._policy_frequency == 0:
            actor_state, qf1_state, actor_loss_value = update_actor(
                self._actor,
                self._qf1,
                batch["observations"],
                self.tau,
            )
            self._actor = actor_state
            self._qf1 = qf1_state
            self._update_info["actor_loss"] = actor_loss_value
        self._update_step += 1
        return self._update_info

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        rng, actions = sample_actions_jit(
            self._rng, self._actor, observations, self._exploration_noise, self._action_low_high
        )
        self._rng = rng
        return np.asarray(actions)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = eval_actions_jit(self._actor, observations)
        return np.asarray(actions)


import os
import glob
import subprocess
import time
import tqdm
import wandb
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from jaxrl3.data import ReplayBuffer
from jaxrl3.evaluation import evaluate

from utils import prepare_experiment_envs

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "HalfCheetah-v4", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 1000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("timesteps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
)
flags.DEFINE_integer("replay_buffer_size", 1000000, "Replay buffer capacity.")
flags.DEFINE_boolean("tqdm", False, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")

flags.DEFINE_string("wrapper_id", "InvertAction", "Action effect wrapper.")
flags.DEFINE_string("wrapper_dim", "0", "Dimension for action effect wrapper setting.")
flags.DEFINE_float("wrapper_value", 0.5, "Value for action effect wrapper setting.")
flags.DEFINE_integer("wrapper_repeat", 4, "Amount of repeat steps for action effect wrapper setting.")
flags.DEFINE_string("exp_type", "basic", "Kind of experiment.")

flags.DEFINE_boolean("wandb", False, "Use wandb.")
flags.DEFINE_boolean("wandb_offline", False, "Use wandb offline mode.")
flags.DEFINE_string("project_name", "jaxrl3", "Project name for wandb.")

config_flags.DEFINE_config_file(
    "config",
    "configs/ddpg_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    if FLAGS.wandb:
        os.makedirs(f"./wandb/{FLAGS.project_name}", exist_ok=True)
        wandb_id = wandb.util.generate_id()
        wandb_run = wandb.init(
            project=FLAGS.project_name,
            config=FLAGS,
            mode="offline" if FLAGS.wandb_offline else "online",
            dir=f"./wandb/{FLAGS.project_name}",
            id=wandb_id
        )
    else:
        wandb_run = None

    env, eval_envs = prepare_experiment_envs(
        exp_type=FLAGS.exp_type,
        wrapper_id=FLAGS.wrapper_id,
        wrapper_dim=FLAGS.wrapper_dim,
        wrapper_value=FLAGS.wrapper_value,
        env_name=FLAGS.env_name,
        seed=FLAGS.seed,
        timesteps=FLAGS.timesteps,
        pixel=False,
        action_repeat=None,
        image_size=None,
        num_stack=None,
        wrapper_repeat=FLAGS.wrapper_repeat,
    )
    if FLAGS.wandb:
        eval_returns = {"timesteps": []}
        for task_key in eval_envs.keys():
            eval_returns[f"return_{task_key}"] = []
        train_returns = {"timesteps": [], "return": []}

    kwargs = dict(FLAGS.config)
    agent = DDPGLearner(FLAGS.seed, env.observation_space, env.action_space, **kwargs)

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.replay_buffer_size
    )
    replay_buffer.seed(FLAGS.seed)

    start_time = time.time()
    observation, _ = env.reset()
    done = False
    for i in tqdm.tqdm(
        range(1, FLAGS.timesteps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if not terminated:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, _ = env.reset()
            done = False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                if wandb_run is not None:
                    wandb_run.log({f"training/{decode[k]}": v}, step=i)
            if wandb_run is not None:
                train_returns["timesteps"].append(i)
                train_returns["return"].append(info["episode"]["r"])

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0 and wandb_run is not None:
                for k, v in update_info.items():
                    wandb_run.log({f"training/{k}": v}, step=i)
                wandb_run.log(
                    {"training/fps": i / (time.time() - start_time)},
                    step=i
                )

        if i % FLAGS.eval_interval == 0:
            if wandb_run is not None:
                for task_key, eval_env in eval_envs.items():
                    eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
                    for k, v in eval_info.items():
                        wandb_run.log({f"evaluation_{task_key}/{k}": v}, step=i)
                    eval_returns[f"return_{task_key}"].append(eval_info["return"])
                eval_returns["timesteps"].append(i)

    if wandb_run is not None:
        wandb_run.finish()

        # save stored train/eval returns
        returns_dir = (
            f"eval/returns/{FLAGS.project_name}/exp_{FLAGS.exp_type}/"
            f"ddpg__{FLAGS.env_name}__steps_{FLAGS.timesteps}__wid_{FLAGS.wrapper_id}__"
            f"wdim_{FLAGS.wrapper_dim}__wval_{FLAGS.wrapper_value}__wrep_{FLAGS.wrapper_repeat}__"
            f"seed_{FLAGS.seed}__{wandb_id}/"
        )
        os.makedirs(returns_dir, exist_ok=True)
        np.savez(f"{returns_dir}/train.npz", **train_returns)
        np.savez(f"{returns_dir}/eval.npz", **eval_returns)

        wandb_dir = glob.glob(f"./wandb/{FLAGS.project_name}/wandb/*{wandb_id}")
        assert len(wandb_dir) == 1, "Wandb id is not unique."
        if FLAGS.wandb_offline:
            assert wandb_id is not None, "Provide wandb_id."
            offline_wandb_dir = glob.glob(f"./wandb/{FLAGS.project_name}/wandb/offline-run*{wandb_id}")
            assert len(offline_wandb_dir) == 1, "Offline wandb id is not unique."
            subprocess.run(["wandb", "sync", "--include-offline", offline_wandb_dir[0]])


if __name__ == "__main__":
    app.run(main)
