import os
import warnings
from typing import Dict, List, Optional, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, EventCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    sync_envs_normalization
)


class MultiEvalCallback(EvalCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_envs: Dict[str, Union[gym.Env, VecEnv]],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 10,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        EventCallback.__init__(self, callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = {k: -np.inf for k in eval_envs.keys()}
        self.last_mean_reward = {k: -np.inf for k in eval_envs.keys()}
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        for k, eval_env in eval_envs.items():
            if not isinstance(eval_env, VecEnv):
                eval_envs[k] = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        self.eval_envs = eval_envs
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = {k: os.path.join(log_path, f"evaluations_{k}") for k in self.eval_envs.keys()}
        self.log_path = log_path
        self.evaluations_results: Dict[str, List[List[float]]] = {
            k: [] for k in self.eval_envs.keys()
        }
        self.evaluations_timesteps: Dict[str, List[List[float]]] = {
            k: [] for k in self.eval_envs.keys()
        }
        self.evaluations_length: Dict[str, List[List[float]]] = {
            k: [] for k in self.eval_envs.keys()
        }
        # For computing success rate
        self._is_success_buffer: List[bool] = []
        self.evaluations_successes: Dict[str, List[List[float]]] = {
            k: [] for k in self.eval_envs.keys()
        }

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        for eval_key, eval_env in self.eval_envs.items():
            if not isinstance(self.training_env, type(eval_env)):
                warnings.warn(
                    f"Training and eval env ({eval_key}) are not of the same type" "{self.training_env} != {eval_env}"
                )

            # Create folders if needed
            if self.log_path[eval_key] is not None:
                os.makedirs(os.path.dirname(self.log_path[eval_key]), exist_ok=True)
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            for eval_key, eval_env in self.eval_envs.items():
                # Sync training and eval env if there is VecNormalize
                if self.model.get_vec_normalize_env() is not None:
                    try:
                        sync_envs_normalization(self.training_env, eval_env)
                    except AttributeError as e:
                        raise AssertionError(
                            "Training and eval env are not wrapped the same way, "
                            "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                            "and warning above."
                        ) from e

                # Reset success rate buffer
                self._is_success_buffer = []

                episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=self.warn,
                    callback=self._log_success_callback,
                )

                if self.log_path is not None:
                    assert isinstance(episode_rewards, list)
                    assert isinstance(episode_lengths, list)
                    self.evaluations_timesteps[eval_key].append(self.num_timesteps)
                    self.evaluations_results[eval_key].append(episode_rewards)
                    self.evaluations_length[eval_key].append(episode_lengths)

                    kwargs = {}
                    # Save success log if present
                    if len(self._is_success_buffer) > 0:
                        self.evaluations_successes[eval_key].append(self._is_success_buffer)
                        kwargs = dict(successes=self.evaluations_successes[eval_key])

                    np.savez(
                        self.log_path[eval_key],
                        timesteps=self.evaluations_timesteps[eval_key],
                        results=self.evaluations_results[eval_key],
                        ep_lengths=self.evaluations_length[eval_key],
                        **kwargs,
                    )

                mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
                mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
                self.last_mean_reward[eval_key] = float(mean_reward)

                if self.verbose >= 1:
                    print(
                        f"Eval task {eval_key} num_timesteps={self.num_timesteps}, ",
                        f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                    )
                    print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                # Add to current Logger
                self.logger.record(f"eval_{eval_key}/mean_reward", float(mean_reward))
                self.logger.record(f"eval_{eval_key}/mean_ep_length", mean_ep_length)

                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    if self.verbose >= 1:
                        print(f"Success rate: {100 * success_rate:.2f}%")
                    self.logger.record(f"eval_{eval_key}/success_rate", success_rate)

                if mean_reward > self.best_mean_reward[eval_key]:
                    if self.verbose >= 1:
                        print("New best mean reward!")
                    if self.best_model_save_path is not None:
                        self.model.save(
                            os.path.join(self.best_model_save_path, f"{eval_key}_best_model")
                        )
                    self.best_mean_reward[eval_key] = float(mean_reward)
                    # Trigger callback on new best model, if needed
                    if self.callback_on_new_best is not None:
                        continue_training = self.callback_on_new_best.on_step()

                # Trigger callback after every evaluation, if needed
                if self.callback is not None:
                    continue_training = continue_training and self._on_event()

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

        return continue_training
