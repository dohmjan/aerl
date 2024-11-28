import gymnasium as gym
from gymnasium.envs.registration import registry

from jaxrl3.wrappers import wrap_gym, wrap_pixels, set_universal_seed
from jaxrl3.wrappers.repeat_action import RepeatAction as RepeatActionJAXRL

from gymnasium.wrappers import ClipAction
from aerl import (
    InvertAction,
    NoiseAction,
    SineNoiseAction,
    OffsetAction,
    ScaleAction,
    RepeatAction,
    ZeroAction,
    SwapAction,
    ContextAwareObservation
)


def get_wrapper_kwargs(wrapper_id, wrapper_dim, wrapper_value):
    try:
        dim = int(wrapper_dim)
    except ValueError:
        dim = None
    if wrapper_id in ["InvertAction", "SwapAction"]:
        wrapper_kwargs = {"dim": dim}
    else:
        wrapper_kwargs = {"dim": dim, "value": wrapper_value}
    return wrapper_kwargs


def check_env_id(env_id):
    dm_control_env_ids = [
        id
        for id in registry
        if id.startswith("dm_control/") and id != "dm_control/compatibility-env-v0"
    ]
    if not env_id.startswith("dm_control/"):
        for id in dm_control_env_ids:
            if env_id in id:
                env_id = "dm_control/" + env_id
    if env_id not in registry:
        raise ValueError("Provide valid env id.")
    return env_id


def make_and_wrap_env(env_id, pixel, image_size, num_stack):
    env_id = check_env_id(env_id)

    if pixel:
        if "quadruped" in env_id:
            camera_id = 2
        else:
            camera_id = 0

        render_kwargs = dict(camera_id=camera_id, height=image_size, width=image_size)
        if env_id.startswith("dm_control"):
            env = gym.make(env_id, render_mode="rgb_array", render_kwargs=render_kwargs)
        else:
            render_kwargs.pop("camera_id")
            env = gym.make(env_id, render_mode="rgb_array", **render_kwargs)

        return wrap_pixels(
            env,
            action_repeat=0,  # 0 to skip ActionRepeat wrapper, it's applied later.
            num_stack=num_stack,
        )
    else:
        env = gym.make(env_id)
        return wrap_gym(env)


def wrap_action_effect(env, wrapper_id, wrapper_kwargs, toggle_at_episode=0):
    if wrapper_id == "InvertAction":
        env = InvertAction(env, toggle_at_episode=toggle_at_episode, **wrapper_kwargs)
    elif wrapper_id == "NoiseAction":
        env = NoiseAction(env, toggle_at_episode=toggle_at_episode, **wrapper_kwargs)
    elif wrapper_id == "SineNoiseAction":
        env = SineNoiseAction(env, toggle_at_episode=toggle_at_episode, **wrapper_kwargs)
    elif wrapper_id == "OffsetAction":
        env = OffsetAction(env, toggle_at_episode=toggle_at_episode, **wrapper_kwargs)
    elif wrapper_id == "ScaleAction":
        env = ScaleAction(env, toggle_at_episode=toggle_at_episode, **wrapper_kwargs)
    elif wrapper_id == "RepeatAction":
        env = RepeatAction(env, toggle_at_episode=toggle_at_episode, **wrapper_kwargs)
    elif wrapper_id == "ZeroAction":
        env = ZeroAction(env, toggle_at_episode=toggle_at_episode, **wrapper_kwargs)
    elif wrapper_id == "SwapAction":
        env = SwapAction(env, toggle_at_episode=toggle_at_episode, **wrapper_kwargs)
    else:
        raise NotImplementedError()
    return env


def wrap_remaining(env, pixel, action_repeat):
    env = ClipAction(env)
    env = DynamicsHintObservation(env)
    if pixel:
        # We count the number of steps in the action effect wrapper. Hence, RepeatAction has to
        # come after, as we want the repeated actions counted.
        env = RepeatActionJAXRL(env, action_repeat=action_repeat)
    return env


def multitask_experiment_setup(
    env_name,
    seed,
    timesteps,
    toggle_at_episode,
    wrapper_id="InvertAction",
    wrapper_kwargs={"dim": 0},
    pixel=False,
    action_repeat=None,
    image_size=None,
    num_stack=None
):
    if pixel:
        assert action_repeat is not None
        assert image_size is not None
        assert num_stack is not None

    eval_env_task_0 = make_and_wrap_env(env_name, pixel, image_size, num_stack)
    set_universal_seed(eval_env_task_0, seed)
    eval_env_task_0 = wrap_action_effect(
        eval_env_task_0,
        wrapper_id,
        wrapper_kwargs,
        toggle_at_episode=timesteps + 1
    )
    eval_env_task_0 = wrap_remaining(eval_env_task_0, pixel, action_repeat)
    eval_env_task_1 = make_and_wrap_env(env_name, pixel, image_size, num_stack)
    set_universal_seed(eval_env_task_1, seed)
    eval_env_task_1 = wrap_action_effect(
        eval_env_task_1,
        wrapper_id,
        wrapper_kwargs,
        toggle_at_episode=0
    )
    eval_env_task_1 = wrap_remaining(eval_env_task_1, pixel, action_repeat)
    eval_envs = {"task_0": eval_env_task_0, "task_1": eval_env_task_1}

    env = make_and_wrap_env(env_name, pixel, image_size, num_stack)
    set_universal_seed(env, seed)

    env = wrap_action_effect(
        env,
        wrapper_id,
        wrapper_kwargs,
        toggle_at_episode=toggle_at_episode
    )
    env = wrap_remaining(env, pixel, action_repeat)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    return env, eval_envs


def get_experiment_wrapper_specs(exp_type, wrapper_id, wrapper_kwargs):
    if exp_type == "basic":
        toggle_at_episode = 0.5
    elif exp_type == "reference":
        toggle_at_episode = 0
    elif exp_type == "reference_task_0":
        toggle_at_episode = 10000000
    else:
        raise NotImplementedError()
    return wrapper_id, wrapper_kwargs, toggle_at_episode


def prepare_experiment_envs(
    exp_type,
    wrapper_id,
    wrapper_dim,
    wrapper_value,
    env_name,
    seed,
    timesteps,
    pixel,
    action_repeat,
    image_size,
    num_stack
):
    wrapper_kwargs = get_wrapper_kwargs(wrapper_id, wrapper_dim, wrapper_value)
    wrapper_id, wrapper_kwargs, toggle_at_episode = get_experiment_wrapper_specs(
        exp_type,
        wrapper_id,
        wrapper_kwargs
    )
    env, eval_envs = multitask_experiment_setup(
        env_name=env_name,
        seed=seed,
        timesteps=timesteps,
        toggle_at_episode=toggle_at_episode,
        wrapper_id=wrapper_id,
        wrapper_kwargs=wrapper_kwargs,
        pixel=pixel,
        action_repeat=action_repeat,
        image_size=image_size,
        num_stack=num_stack
    )
    return env, eval_envs

