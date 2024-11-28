from stable_baselines3 import SAC, PPO, HerReplayBuffer
from stable_baselines3.common.callbacks import CallbackList

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, ClipAction, RescaleAction

from aerl import (
    InvertAction,
    NoiseAction,
    SineNoiseAction,
    OffsetAction,
    ScaleAction,
    RepeatAction,
    ZeroAction,
    SwapAction
)

from utils import MultiEvalCallback


def make_env(env_id, seed=0):
    env = gym.make(env_id)
    if "dm_control" in env.spec.id:
        env = FlattenObservation(env)
    env = RescaleAction(env, -1, 1)
    env = ClipAction(env)

    # set universal seed
    _, _ = env.reset(seed=seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)

    return env


def wrap_action_effect(env, wrapper_id, wrapper_kwargs, toggle_at_step=0):
    if wrapper_id == "InvertAction":
        env = InvertAction(env, toggle_at_step=toggle_at_step, **wrapper_kwargs)
    elif wrapper_id == "NoiseAction":
        env = NoiseAction(env, toggle_at_step=toggle_at_step, **wrapper_kwargs)
    elif wrapper_id == "SineNoiseAction":
        env = SineNoiseAction(env, toggle_at_step=toggle_at_step, **wrapper_kwargs)
    elif wrapper_id == "OffsetAction":
        env = OffsetAction(env, toggle_at_step=toggle_at_step, **wrapper_kwargs)
    elif wrapper_id == "ScaleAction":
        env = ScaleAction(env, toggle_at_step=toggle_at_step, **wrapper_kwargs)
    elif wrapper_id == "RepeatAction":
        env = RepeatAction(env, toggle_at_step=toggle_at_step, **wrapper_kwargs)
    elif wrapper_id == "ZeroAction":
        env = ZeroAction(env, toggle_at_step=toggle_at_step, **wrapper_kwargs)
    elif wrapper_id == "SwapAction":
        env = SwapAction(env, toggle_at_step=toggle_at_step, **wrapper_kwargs)
    else:
        raise NotImplementedError()
    return env


def wrap_remaining(env):
    env = ClipAction(env)
    return env


def sequential_experiment_setup(
    log_path,
    seed,
    env_id,
    timesteps,
    wrapper_id="InvertAction",
    wrapper_kwargs={"dim": 0},
    n_segments=2,
):
    if wrapper_id in ["InvertAction", "SwapAction"]:
        wrapper_kwargs.pop("value")

    eval_env_task_0 = make_env(env_id, seed)
    eval_env_task_0 = wrap_remaining(eval_env_task_0)
    eval_env_task_1 = make_env(env_id, seed)
    eval_env_task_1 = wrap_action_effect(eval_env_task_1, wrapper_id, wrapper_kwargs, toggle_at_step=0)
    eval_env_task_1 = wrap_remaining(eval_env_task_1)
    eval_envs = {"task_0": eval_env_task_0, "task_1": eval_env_task_1}

    eval_callback = MultiEvalCallback(eval_envs, eval_freq=1000, log_path=log_path, verbose=0)

    toggle_at_step = [switch * timesteps // n_segments for switch in range(1, n_segments)]

    env = make_env(env_id, seed)
    env = wrap_action_effect(
        env,
        wrapper_id,
        wrapper_kwargs,
        toggle_at_step=toggle_at_step
    )
    env = wrap_remaining(env)
    return env, eval_callback


def sequential_multiple_experiment_setup(
    log_path,
    seed,
    env_id,
    timesteps,
    wrapper_sequence=["InvertAction"],
    wrapper_sequence_kwargs=[{"dim": 0}],
):
    eval_env_task_0 = make_env(env_id, seed)
    eval_env_task_0 = wrap_remaining(eval_env_task_0)
    eval_envs = {"task_0": eval_env_task_0}
    for i, (wrapper_id, wrapper_kwargs) in enumerate(zip(wrapper_sequence, wrapper_sequence_kwargs)):
        if wrapper_id in ["InvertAction", "SwapAction"]:
            wrapper_kwargs.pop("value")

        eval_env_task_i = make_env(env_id, seed)
        eval_env_task_i = wrap_action_effect(eval_env_task_i, wrapper_id, wrapper_kwargs, toggle_at_step=0)
        eval_env_task_i = wrap_remaining(eval_env_task_i)
        eval_envs[f"task_{i+1}"] = eval_env_task_i

    eval_callback = MultiEvalCallback(eval_envs, eval_freq=1000, log_path=log_path, verbose=0)

    n_segments = len(wrapper_sequence) + 1

    env = make_env(env_id)
    for i, (wrapper_id, wrapper_kwargs) in enumerate(zip(wrapper_sequence, wrapper_sequence_kwargs)):
        if wrapper_id in ["InvertAction", "SwapAction"]:
            wrapper_kwargs.pop("value")

        env = wrap_action_effect(
            env,
            wrapper_id,
            wrapper_kwargs,
            toggle_at_step=timesteps // n_segments * (i + 1)
        )
    env = wrap_remaining(env)
    return env, eval_callback


def sac_gymnasium(
    timesteps=2000000,
    seed=0,
    wrapper_id="InvertAction",
    wrapper_kwargs={"dim": 0},
    wandb_run=None,
    project_name=""
):
    env_id = "HalfCheetah-v4"
    exp_id = "sac_gymnasium"

    if wandb_run is not None:
        log_path = (
            f"eval/returns/{project_name}/exp_{exp_id}/sb3_sac__{env_id.split('/')[-1]}__"
            f"steps_{timesteps}__wid_{wrapper_id}__wdim_{wrapper_kwargs['dim']}__"
            f"wval_{wrapper_kwargs['value']}__"
            f"seed_{seed}__{wandb_run.id}/"
        )

    env, eval_callback = sequential_experiment_setup(
        log_path=log_path,
        env_id=env_id,
        timesteps=timesteps,
        wrapper_id=wrapper_id,
        wrapper_kwargs=wrapper_kwargs,
        seed=seed,
    )
    if wandb_run is not None:
        tb_log = f"{log_path}/{wandb_run.id}"
        callback = CallbackList([eval_callback, WandbCallback()])
    else:
        tb_log = None
        callback = eval_callback

    model = SAC("MlpPolicy", env, seed=seed, stats_window_size=1, tensorboard_log=tb_log)
    model.learn(total_timesteps=timesteps, callback=callback)


def sac_dmc(
    timesteps=1000000,
    seed=0,
    wrapper_id="InvertAction",
    wrapper_kwargs={"dim": 0},
    wandb_run=None,
    project_name=""
):
    env_id = "dm_control/walker-walk-v0"
    exp_id = "sac_dmc"

    if wandb_run is not None:
        log_path = (
            f"eval/returns/{project_name}/exp_{exp_id}/sb3_sac__{env_id.split('/')[-1]}__"
            f"steps_{timesteps}__wid_{wrapper_id}__wdim_{wrapper_kwargs['dim']}__"
            f"wval_{wrapper_kwargs['value']}__"
            f"seed_{seed}__{wandb_run.id}/"
        )

    env, eval_callback = sequential_experiment_setup(
        log_path=log_path,
        env_id=env_id,
        timesteps=timesteps,
        wrapper_id=wrapper_id,
        wrapper_kwargs=wrapper_kwargs,
        seed=seed,
    )
    if wandb_run is not None:
        tb_log = f"{log_path}/{wandb_run.id}"
        callback = CallbackList([eval_callback, WandbCallback()])
    else:
        tb_log = None
        callback = eval_callback
    model = SAC("MlpPolicy", env, seed=seed, stats_window_size=1, tensorboard_log=tb_log)
    model.learn(total_timesteps=timesteps, callback=callback)


def ppo_dmc(
    timesteps=1000000,
    seed=0,
    wrapper_id="InvertAction",
    wrapper_kwargs={"dim": 0},
    wandb_run=None,
    project_name=""
):
    env_id = "dm_control/walker-walk-v0"
    exp_id = "ppo_dmc"

    if wandb_run is not None:
        log_path = (
            f"eval/returns/{project_name}/exp_{exp_id}/sb3_ppo__{env_id.split('/')[-1]}__"
            f"steps_{timesteps}__wid_{wrapper_id}__wdim_{wrapper_kwargs['dim']}__"
            f"wval_{wrapper_kwargs['value']}__"
            f"seed_{seed}__{wandb_run.id}/"
        )

    env, eval_callback = sequential_experiment_setup(
        log_path=log_path,
        env_id=env_id,
        timesteps=timesteps,
        wrapper_id=wrapper_id,
        wrapper_kwargs=wrapper_kwargs,
        seed=seed,
    )
    if wandb_run is not None:
        tb_log = f"{log_path}/{wandb_run.id}"
        callback = CallbackList([eval_callback, WandbCallback()])
    else:
        tb_log = None
        callback = eval_callback
    model = PPO("MlpPolicy", env, seed=seed, stats_window_size=1, tensorboard_log=tb_log)
    model.learn(total_timesteps=timesteps, callback=callback)


def sac_goal(
    timesteps=200000,
    seed=0,
    wrapper_id="InvertAction",
    wrapper_kwargs={"dim": 0},
    wandb_run=None,
    project_name="",
    env_name="FetchReach-v2"
):
    exp_id = "basic"

    if wandb_run is not None:
        log_path = (
            f"eval/returns/{project_name}/exp_{exp_id}/sb3_sac__{env_name.split('/')[-1]}__"
            f"steps_{timesteps}__wid_{wrapper_id}__wdim_{wrapper_kwargs['dim']}__"
            f"wval_{wrapper_kwargs['value']}__"
            f"seed_{seed}__{wandb_run.id}/"
        )

    env, eval_callback = sequential_experiment_setup(
        log_path=log_path,
        env_id=env_name,
        timesteps=timesteps,
        wrapper_id=wrapper_id,
        wrapper_kwargs=wrapper_kwargs,
        seed=seed,
    )
    if wandb_run is not None:
        tb_log = f"{log_path}/{wandb_run.id}"
        callback = CallbackList([eval_callback, WandbCallback()])
    else:
        tb_log = None
        callback = eval_callback

    if env_name == "FetchReach-v2":
        policy_kwargs = dict(n_critics=2, net_arch=[64, 64])
    elif env_name == "FetchPush-v2":
        policy_kwargs = dict(n_critics=2, net_arch=[256, 256, 256])
    else:
        policy_kwargs = None

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        seed=seed,
        stats_window_size=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_log
    )
    model.learn(total_timesteps=timesteps, callback=callback)


def sac_goal_long(
    timesteps=1000000,
    n_segments=10,
    seed=0,
    wrapper_id="InvertAction",
    wrapper_kwargs={"dim": 0},
    wandb_run=None,
    project_name=""
):
    env_id = "FetchReach-v2"
    exp_id = "sac_goal_long"

    if wandb_run is not None:
        log_path = (
            f"eval/returns/{project_name}/exp_{exp_id}/sb3_sac__{env_id.split('/')[-1]}__"
            f"steps_{timesteps}__wid_{wrapper_id}__wdim_{wrapper_kwargs['dim']}__"
            f"wval_{wrapper_kwargs['value']}__"
            f"seed_{seed}__{wandb_run.id}/"
        )

    env, eval_callback = sequential_experiment_setup(
        log_path=log_path,
        env_id=env_id,
        timesteps=timesteps,
        wrapper_id=wrapper_id,
        wrapper_kwargs=wrapper_kwargs,
        n_segments=n_segments,
        seed=seed,
    )
    if wandb_run is not None:
        tb_log = f"{log_path}/{wandb_run.id}"
        callback = CallbackList([eval_callback, WandbCallback()])
    else:
        tb_log = None
        callback = eval_callback
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        seed=seed,
        stats_window_size=1,
        tensorboard_log=tb_log
    )
    model.learn(total_timesteps=timesteps, callback=callback)


def sac_dmc_continual_offset(timesteps=5000000, seed=0, wandb_run=None, project_name=""):
    env_id = "dm_control/walker-walk-v0"
    exp_id = "sac_dmc_continual"
    wrapper_id = "OffsetAction"

    if wandb_run is not None:
        log_path = (
            f"eval/returns/{project_name}/exp_{exp_id}/sb3_sac__{env_id.split('/')[-1]}__"
            f"steps_{timesteps}__wid_{wrapper_id}__wdim_{0}__wval_{0.5}__"
            f"seed_{seed}__{wandb_run.id}/"
        )

    env, eval_callback = sequential_multiple_experiment_setup(
        log_path=log_path,
        env_id=env_id,
        timesteps=timesteps,
        wrapper_sequence=["OffsetAction"] * 9,
        wrapper_sequence_kwargs=[
            {"dim": 0, "value": -0.1},
            {"dim": 0, "value": -0.2},
            {"dim": 0, "value": -0.3},
            {"dim": 0, "value": -0.4},
            {"dim": 0, "value": -0.5},
            {"dim": 0, "value": -0.6},
            {"dim": 0, "value": -0.7},
            {"dim": 0, "value": -0.8},
            {"dim": 0, "value": -0.9},
        ],
        seed=seed,
    )
    if wandb_run is not None:
        tb_log = f"{log_path}/{wandb_run.id}"
        callback = CallbackList([eval_callback, WandbCallback()])
    else:
        tb_log = None
        callback = eval_callback
    model = SAC("MlpPolicy", env, seed=seed, stats_window_size=1, tensorboard_log=tb_log)
    model.learn(total_timesteps=timesteps, callback=callback)


def sac_dmc_continual_noise(timesteps=5000000, seed=0, wandb_run=None, project_name=""):
    env_id = "dm_control/walker-walk-v0"
    exp_id = "sac_dmc_continual"
    wrapper_id = "NoiseAction"

    if wandb_run is not None:
        log_path = (
            f"eval/returns/{project_name}/exp_{exp_id}/sb3_sac__{env_id.split('/')[-1]}__"
            f"steps_{timesteps}__wid_{wrapper_id}__wdim_{0}__wval_{0.5}__"
            f"seed_{seed}__{wandb_run.id}/"
        )

    env, eval_callback = sequential_multiple_experiment_setup(
        log_path=log_path,
        env_id=env_id,
        timesteps=timesteps,
        wrapper_sequence=["NoiseAction"] * 9,
        wrapper_sequence_kwargs=[
            {"dim": 0, "value": 0.1},
            {"dim": 0, "value": 0.2},
            {"dim": 0, "value": 0.3},
            {"dim": 0, "value": 0.4},
            {"dim": 0, "value": 0.5},
            {"dim": 0, "value": 0.6},
            {"dim": 0, "value": 0.7},
            {"dim": 0, "value": 0.8},
            {"dim": 0, "value": 0.9},
        ],
        seed=seed,
    )
    if wandb_run is not None:
        tb_log = f"{log_path}/{wandb_run.id}"
        callback = CallbackList([eval_callback, WandbCallback()])
    else:
        tb_log = None
        callback = eval_callback
    model = SAC("MlpPolicy", env, seed=seed, stats_window_size=1, tensorboard_log=tb_log)
    model.learn(total_timesteps=timesteps, callback=callback)


def sac_dmc_continual_scale(timesteps=5000000, seed=0, wandb_run=None, project_name=""):
    env_id = "dm_control/walker-walk-v0"
    exp_id = "sac_dmc_continual"
    wrapper_id = "ScaleAction"

    if wandb_run is not None:
        log_path = (
            f"eval/returns/{project_name}/exp_{exp_id}/sb3_sac__{env_id.split('/')[-1]}__"
            f"steps_{timesteps}__wid_{wrapper_id}__wdim_{0}__wval_{0.5}__"
            f"seed_{seed}__{wandb_run.id}/"
        )

    env, eval_callback = sequential_multiple_experiment_setup(
        log_path=log_path,
        env_id=env_id,
        timesteps=timesteps,
        wrapper_sequence=["ScaleAction"] * 9,
        wrapper_sequence_kwargs=[
            {"dim": 0, "value": 0.9},
            {"dim": 0, "value": 0.8},
            {"dim": 0, "value": 0.7},
            {"dim": 0, "value": 0.6},
            {"dim": 0, "value": 0.5},
            {"dim": 0, "value": 0.4},
            {"dim": 0, "value": 0.3},
            {"dim": 0, "value": 0.2},
            {"dim": 0, "value": 0.1},
        ],
        seed=seed,
    )
    if wandb_run is not None:
        tb_log = f"{log_path}/{wandb_run.id}"
        callback = CallbackList([eval_callback, WandbCallback()])
    else:
        tb_log = None
        callback = eval_callback
    model = SAC("MlpPolicy", env, seed=seed, stats_window_size=1, tensorboard_log=tb_log)
    model.learn(total_timesteps=timesteps, callback=callback)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="seed", default=0)
    parser.add_argument('--exp', type=str, help="experiment setting", default="basic")
    parser.add_argument('--wrapper_id', type=str, help="wrapper name", default="InvertAction")
    parser.add_argument('--wrapper_value', type=float, help="value corresponding to wrapper setting", default=0.3)
    parser.add_argument('--wrapper_dim', help="dim corresponding to wrapper setting", default=0)
    parser.add_argument('--wandb', action='store_true', help="use wandb", default=False)
    parser.add_argument('--wandb_offline', action='store_true', help="wandb offline mode", default=False)
    parser.add_argument('--project_name', type=str, help="wandb project name", default="ds_seq")
    parser.add_argument('--env_name', type=str, help="env for sac_goal exp", default="FetchReach-v2")
    parser.add_argument('--timesteps', type=int, help="num timesteps", default=200000)
    args = parser.parse_args()

    if args.wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
        except ImportError as e:
            print("Make sure wandb and tensorboard is installed. (pip install wandb tensorboard)")
            raise e
        wandb_id = wandb.util.generate_id()
        run = wandb.init(
            project=args.project_name,
            config=vars(args),
            sync_tensorboard=True,
            mode="offline" if args.wandb_offline else "online",
            id=wandb_id
        )
    else:
        run = None

    try:
        dim = int(args.wrapper_dim)
    except ValueError:
        dim = None
    wrapper_kwargs = {"dim": dim, "value": args.wrapper_value}

    if args.exp == "sac_gymnasium":
        sac_gymnasium(seed=args.seed, wandb_run=run, wrapper_id=args.wrapper_id, wrapper_kwargs=wrapper_kwargs, project_name=args.project_name)
    elif args.exp == "sac_dmc":
        sac_dmc(seed=args.seed, wandb_run=run, wrapper_id=args.wrapper_id, wrapper_kwargs=wrapper_kwargs, project_name=args.project_name)
    elif args.exp == "ppo_dmc":
        ppo_dmc(seed=args.seed, wandb_run=run, wrapper_id=args.wrapper_id, wrapper_kwargs=wrapper_kwargs, project_name=args.project_name)
    elif args.exp == "basic":
        sac_goal(seed=args.seed, wandb_run=run, wrapper_id=args.wrapper_id, wrapper_kwargs=wrapper_kwargs, project_name=args.project_name, env_name=args.env_name, timesteps=args.timesteps)

    elif args.exp == "sac_goal_long":
        sac_goal_long(seed=args.seed, wandb_run=run, wrapper_id=args.wrapper_id, wrapper_kwargs=wrapper_kwargs, project_name=args.project_name)

    elif args.exp == "sac_dmc_continual_offset":
        sac_dmc_continual_offset(seed=args.seed, wandb_run=run, project_name=args.project_name)
    elif args.exp == "sac_dmc_continual_noise":
        sac_dmc_continual_noise(seed=args.seed, wandb_run=run, project_name=args.project_name)
    elif args.exp == "sac_dmc_continual_scale":
        sac_dmc_continual_scale(seed=args.seed, wandb_run=run, project_name=args.project_name)

    else:
        raise NotImplementedError()

    if run is not None:
        run.finish()
        if args.wandb_offline:
            import subprocess
            import glob

            assert wandb_id is not None, "Provide wandb_id."
            offline_wandb_dir = glob.glob(f"./wandb/offline-run*{wandb_id}")
            assert len(offline_wandb_dir) == 1, "Offline wandb id is not unique."
            subprocess.run(["wandb", "sync", "--include-offline", offline_wandb_dir[0]])
