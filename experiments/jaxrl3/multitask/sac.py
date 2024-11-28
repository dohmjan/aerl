#! /usr/bin/env python
import os
import glob
import subprocess
import time
import tqdm
import wandb
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from jaxrl3.agents import SACLearner
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
flags.DEFINE_string("exp_type", "basic", "Kind of experiment.")

flags.DEFINE_boolean("wandb", False, "Use wandb.")
flags.DEFINE_boolean("wandb_offline", False, "Use wandb offline mode.")
flags.DEFINE_string("project_name", "jaxrl3", "Project name for wandb.")

config_flags.DEFINE_config_file(
    "config",
    "configs/sac_default.py",
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
        num_stack=None
    )
    if FLAGS.wandb:
        eval_returns = {"timesteps": []}
        for task_key in eval_envs.keys():
            eval_returns[f"return_{task_key}"] = []
        train_returns = {"timesteps": [], "return": []}

    kwargs = dict(FLAGS.config)
    agent = SACLearner(FLAGS.seed, env.observation_space, env.action_space, **kwargs)

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
        if done:
            pass

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
            f"sac__{FLAGS.env_name}__steps_{FLAGS.timesteps}__wid_{FLAGS.wrapper_id}__"
            f"wdim_{FLAGS.wrapper_dim}__wval_{FLAGS.wrapper_value}__seed_{FLAGS.seed}__{wandb_id}/"
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
