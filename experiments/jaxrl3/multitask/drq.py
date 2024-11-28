#! /usr/bin/env python
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import glob
import subprocess
import time
import pickle

import tqdm
import wandb
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from jaxrl3.agents import DrQLearner
from jaxrl3.data import MemoryEfficientReplayBuffer
from jaxrl3.evaluation import evaluate

from utils import prepare_experiment_envs

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 1000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("timesteps", 1000000, "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
)
flags.DEFINE_integer("image_size", 64, "Image size.")
flags.DEFINE_integer("num_stack", 3, "Stack frames.")
flags.DEFINE_integer("replay_buffer_size", 100000, "Replay buffer capacity.")
flags.DEFINE_integer(
    "action_repeat", None, "Action repeat, if None, uses 2 or PlaNet default values."
)
flags.DEFINE_boolean("tqdm", False, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("save_buffer", False, "Save the replay buffer.")

flags.DEFINE_string("wrapper_id", "InvertAction", "Action effect wrapper.")
flags.DEFINE_string("wrapper_dim", "0", "Dimension for action effect wrapper setting.")
flags.DEFINE_float("wrapper_value", 0.5, "Value for action effect wrapper setting.")
flags.DEFINE_string("exp_type", "basic", "Kind of experiment.")

flags.DEFINE_boolean("wandb", False, "Use wandb.")
flags.DEFINE_boolean("wandb_offline", False, "Use wandb offline mode.")
flags.DEFINE_string("project_name", "jaxrl3", "Project name for wandb.")

config_flags.DEFINE_config_file(
    "config",
    "configs/drq_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

PLANET_ACTION_REPEAT = {
    "cartpole-swingup-v0": 8,
    "reacher-easy-v0": 4,
    "cheetah-run-v0": 4,
    "finger-spi-n-0": 2,
    "ball_in_cup-catch-v0": 4,
    "walker-walk-v0": 2,
}


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

    if FLAGS.action_repeat is not None:
        action_repeat = FLAGS.action_repeat
    else:
        action_repeat = PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)

    env, eval_envs = prepare_experiment_envs(
        exp_type=FLAGS.exp_type,
        wrapper_id=FLAGS.wrapper_id,
        wrapper_dim=FLAGS.wrapper_dim,
        wrapper_value=FLAGS.wrapper_value,
        env_name=FLAGS.env_name,
        seed=FLAGS.seed,
        timesteps=FLAGS.timesteps,
        pixel=True,
        action_repeat=action_repeat,
        image_size=FLAGS.image_size,
        num_stack=FLAGS.num_stack
    )
    if FLAGS.wandb:
        eval_returns = {"timesteps": []}
        for task_key in eval_envs.keys():
            eval_returns[f"return_{task_key}"] = []
        train_returns = {"timesteps": [], "return": []}

    kwargs = dict(FLAGS.config)
    agent = DrQLearner(
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(), **kwargs
    )
    replay_buffer_size = FLAGS.replay_buffer_size or FLAGS.timesteps // action_repeat
    replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, replay_buffer_size
    )
    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size, "include_pixels": False}
    )

    start_time = time.time()
    observation, _ = env.reset()
    done = False
    for i in tqdm.tqdm(
        range(1, FLAGS.timesteps // action_repeat + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
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
                    wandb_run.log({f"training/{decode[k]}": v}, step=i * action_repeat)
            if wandb_run is not None:
                train_returns["timesteps"].append(i)
                train_returns["return"].append(info["episode"]["r"])

        if i >= FLAGS.start_training:
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0 and wandb_run is not None:
                for k, v in update_info.items():
                    wandb_run.log({f"training/{k}": v}, step=i * action_repeat)
                wandb_run.log(
                    {"training/fps": i * action_repeat / (time.time() - start_time)},
                    step=i * action_repeat
                )

        if i % FLAGS.eval_interval == 0:
            if FLAGS.save_buffer:
                dataset_folder = os.path.join("datasets")
                os.makedirs("datasets", exist_ok=True)
                dataset_file = os.path.join(dataset_folder, f"{FLAGS.env_name}")
                with open(dataset_file, "wb") as f:
                    pickle.dump(replay_buffer, f)

            if wandb_run is not None:
                for task_key, eval_env in eval_envs.items():
                    eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
                    for k, v in eval_info.items():
                        wandb_run.log({f"evaluation_{task_key}/{k}": v}, step=i * action_repeat)
                    eval_returns[f"return_{task_key}"].append(eval_info["return"])
                eval_returns["timesteps"].append(i)

    if wandb_run is not None:
        wandb_run.finish()

        # save stored train/eval returns
        returns_dir = (
            f"eval/returns/{FLAGS.project_name}/exp_{FLAGS.exp_type}/"
            f"drq__{FLAGS.env_name}__steps_{FLAGS.timesteps}__wid_{FLAGS.wrapper_id}__"
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

