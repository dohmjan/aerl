import numpy as np
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
    SwapAction
)


def get_wrapper_kwargs(wrapper_id, wrapper_dim, wrapper_value, wrapper_repeat):
    try:
        dim = int(wrapper_dim)
    except ValueError:
        dim = None
    if wrapper_id in ["InvertAction", "SwapAction"]:
        wrapper_kwargs = {"dim": dim}
    elif wrapper_id in ["ZeroAction", "RepeatAction"] and wrapper_repeat is not None:
        wrapper_kwargs = {"dim": dim, "value": wrapper_value, "num_repeat": wrapper_repeat}
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


def wrap_remaining(env, pixel, action_repeat):
    env = ClipAction(env)
    if pixel:
        # We count the number of steps in the action effect wrapper. Hence, RepeatAction has to
        # come after, as we want the repeated actions counted.
        env = RepeatActionJAXRL(env, action_repeat=action_repeat)
    return env


def sequential_experiment_setup(
    env_name,
    seed,
    timesteps,
    wrapper_id="InvertAction",
    wrapper_kwargs={"dim": 0},
    n_segments=2,
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
    eval_env_task_0 = wrap_remaining(eval_env_task_0, pixel, action_repeat)
    eval_env_task_1 = make_and_wrap_env(env_name, pixel, image_size, num_stack)
    set_universal_seed(eval_env_task_1, seed)
    eval_env_task_1 = wrap_action_effect(eval_env_task_1, wrapper_id, wrapper_kwargs, toggle_at_step=0)
    eval_env_task_1 = wrap_remaining(eval_env_task_1, pixel, action_repeat)
    eval_envs = {"task_0": eval_env_task_0, "task_1": eval_env_task_1}

    toggle_at_step = [switch * timesteps // n_segments for switch in range(1, n_segments)]

    env = make_and_wrap_env(env_name, pixel, image_size, num_stack)
    set_universal_seed(env, seed)

    env = wrap_action_effect(
        env,
        wrapper_id,
        wrapper_kwargs,
        toggle_at_step=toggle_at_step
    )
    env = wrap_remaining(env, pixel, action_repeat)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    return env, eval_envs


def sequential_robust_experiment_setup(
    env_name,
    seed,
    timesteps,
    wrapper_id="InvertAction",
    wrapper_kwargs={"dim": 0},
    n_segments=2,
    pixel=False,
    action_repeat=None,
    image_size=None,
    num_stack=None,
    num_contexts=1
):
    if num_contexts == 1:
        return sequential_experiment_setup(
            env_name,
            seed,
            timesteps,
            wrapper_id,
            wrapper_kwargs,
            n_segments,
            pixel,
            action_repeat,
            image_size,
            num_stack,

        )
    if pixel:
        assert action_repeat is not None
        assert image_size is not None
        assert num_stack is not None

    eval_env_task_0 = make_and_wrap_env(env_name, pixel, image_size, num_stack)
    set_universal_seed(eval_env_task_0, seed)
    eval_env_task_0 = wrap_remaining(eval_env_task_0, pixel, action_repeat)
    eval_env_task_1 = make_and_wrap_env(env_name, pixel, image_size, num_stack)
    set_universal_seed(eval_env_task_1, seed)
    eval_env_task_1 = wrap_action_effect(eval_env_task_1, wrapper_id, wrapper_kwargs, toggle_at_step=0)
    eval_env_task_1 = wrap_remaining(eval_env_task_1, pixel, action_repeat)
    eval_envs = {"task_0": eval_env_task_0, "task_1": eval_env_task_1}

    toggle_at_step = [switch * timesteps // n_segments for switch in range(1, n_segments)]

    env = make_and_wrap_env(env_name, pixel, image_size, num_stack)
    set_universal_seed(env, seed)

    env = wrap_action_effect(
        env,
        wrapper_id,
        wrapper_kwargs,
        toggle_at_step=toggle_at_step
    )

    assert len(toggle_at_step) == 1
    toggle_at_steps_robust = []
    for switch in range(toggle_at_step[0] // 2 // env.spec.max_episode_steps * 2):
        toggle_step = switch * env.spec.max_episode_steps
        if switch % 2 != 0:
            toggle_step -= 1
        toggle_at_steps_robust.append(toggle_step)

    assert 1 < num_contexts <= 9
    # # Make chunks of 2 (toggle on + toggle off)
    # chunks = [toggle_at_steps_robust[i:i + 2] for i in range(0, len(toggle_at_steps_robust), 2)]
    # # Distribute chunks evenly across non-default contexts (num_contexts-1)
    # chunks_per_context = [chunks[i::num_contexts - 1] for i in range(num_contexts - 1)]

    def generate_chunks(n, m, d):
        result = []
        step = d + 1  # Distance between start and end for each pair
        total_pairs = (m + 1) // step // n  # Total pairs per sublist

        for i in range(n):
            sublist = []
            for j in range(total_pairs):
                # Calculate start and end for the current pair
                start = i * step + j * step * n
                end = start + d
                sublist.append(start)
                sublist.append(end)
            result.append(sublist)

        return result
    chunks_per_context = generate_chunks(num_contexts, timesteps // 2, env.spec.max_episode_steps - 1)

    assert "value" in wrapper_kwargs.keys()
    wrapper_kwargs_robust = wrapper_kwargs
    value = 1.0
    for c in range(1, num_contexts):
        value -= 0.1

        wrapper_kwargs_robust["value"] = value
        env = wrap_action_effect(
            env,
            wrapper_id,
            wrapper_kwargs_robust,
            # toggle_at_step=np.concatenate(chunks_per_context[c]).tolist()
            toggle_at_step=chunks_per_context[c]
        )
    env = wrap_remaining(env, pixel, action_repeat)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    return env, eval_envs


def sequential_parallel_experiment_setup(
    env_name,
    seed,
    timesteps,
    wrapper_sequence=["InvertAction"],
    wrapper_sequence_kwargs=[{"dim": 0}],
    n_segments=2,
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
    eval_env_task_0 = wrap_remaining(eval_env_task_0, pixel, action_repeat)
    eval_env_task_1 = make_and_wrap_env(env_name, pixel, image_size, num_stack)
    set_universal_seed(eval_env_task_1, seed)
    for i, (wrapper_id, wrapper_kwargs) in enumerate(zip(wrapper_sequence, wrapper_sequence_kwargs)):
        eval_env_task_1 = wrap_action_effect(eval_env_task_1, wrapper_id, wrapper_kwargs, toggle_at_step=0)
    eval_env_task_1 = wrap_remaining(eval_env_task_1, pixel, action_repeat)
    eval_envs = {"task_0": eval_env_task_0, "task_1": eval_env_task_1}

    toggle_at_step = [switch * timesteps // n_segments for switch in range(1, n_segments)]

    env = make_and_wrap_env(env_name, pixel, image_size, num_stack)
    set_universal_seed(env, seed)

    for i, (wrapper_id, wrapper_kwargs) in enumerate(zip(wrapper_sequence, wrapper_sequence_kwargs)):
        env = wrap_action_effect(
            env,
            wrapper_id,
            wrapper_kwargs,
            toggle_at_step=toggle_at_step
        )
    env = wrap_remaining(env, pixel, action_repeat)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    return env, eval_envs


def sequential_multiple_experiment_setup(
    env_name,
    seed,
    timesteps,
    wrapper_sequence=["InvertAction"],
    wrapper_sequence_kwargs=[{"dim": 0}],
    pixel=False,
    action_repeat=None,
    image_size=None,
    num_stack=None
):
    if pixel:
        assert action_repeat is not None
        assert image_size is not None
        assert num_stack is not None

    n_segments = len(wrapper_sequence) + 1

    eval_env_task_0 = make_and_wrap_env(env_name, pixel, image_size, num_stack)
    set_universal_seed(eval_env_task_0, seed)
    eval_env_task_0 = wrap_remaining(eval_env_task_0, pixel, action_repeat)
    eval_envs = {"task_0": eval_env_task_0}
    for i, (wrapper_id, wrapper_kwargs) in enumerate(zip(wrapper_sequence, wrapper_sequence_kwargs)):
        if n_segments > 10:
            if i not in np.linspace((n_segments - 2) / 9, (n_segments - 2), 9, dtype=int):
                continue
        eval_env_task_i = make_and_wrap_env(env_name, pixel, image_size, num_stack)
        set_universal_seed(eval_env_task_0, seed)
        eval_env_task_i = wrap_action_effect(eval_env_task_i, wrapper_id, wrapper_kwargs, toggle_at_step=0)
        eval_env_task_i = wrap_remaining(eval_env_task_i, pixel, action_repeat)
        eval_envs[f"task_{i+1}"] = eval_env_task_i

    env = make_and_wrap_env(env_name, pixel, image_size, num_stack)
    set_universal_seed(env, seed)
    for i, (wrapper_id, wrapper_kwargs) in enumerate(zip(wrapper_sequence, wrapper_sequence_kwargs)):
        env = wrap_action_effect(
            env,
            wrapper_id,
            wrapper_kwargs,
            toggle_at_step=[timesteps // n_segments * (i + 1), timesteps // n_segments * (i + 2)]
        )
    env = wrap_remaining(env, pixel, action_repeat)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    return env, eval_envs


def get_experiment_wrapper_specs(exp_type, wrapper_id, wrapper_kwargs):
    if exp_type in ["basic", "basic_robust"]:
        n_segments = 2
    elif exp_type == "repeated":
        n_segments = 10
    elif exp_type == "continual":
        if wrapper_id == "OffsetAction":
            wrapper_kwargs = [
                {"dim": wrapper_kwargs["dim"], "value": -0.1},
                {"dim": wrapper_kwargs["dim"], "value": -0.2},
                {"dim": wrapper_kwargs["dim"], "value": -0.3},
                {"dim": wrapper_kwargs["dim"], "value": -0.4},
                {"dim": wrapper_kwargs["dim"], "value": -0.5},
                {"dim": wrapper_kwargs["dim"], "value": -0.6},
                {"dim": wrapper_kwargs["dim"], "value": -0.7},
                {"dim": wrapper_kwargs["dim"], "value": -0.8},
                {"dim": wrapper_kwargs["dim"], "value": -0.9},
            ]
        elif wrapper_id == "ScaleAction":
            wrapper_kwargs = [
                {"dim": wrapper_kwargs["dim"], "value": 0.9},
                {"dim": wrapper_kwargs["dim"], "value": 0.8},
                {"dim": wrapper_kwargs["dim"], "value": 0.7},
                {"dim": wrapper_kwargs["dim"], "value": 0.6},
                {"dim": wrapper_kwargs["dim"], "value": 0.5},
                {"dim": wrapper_kwargs["dim"], "value": 0.4},
                {"dim": wrapper_kwargs["dim"], "value": 0.3},
                {"dim": wrapper_kwargs["dim"], "value": 0.2},
                {"dim": wrapper_kwargs["dim"], "value": 0.1},
            ]
        elif wrapper_id == "NoiseAction":
            wrapper_kwargs = [
                {"dim": wrapper_kwargs["dim"], "value": 0.1},
                {"dim": wrapper_kwargs["dim"], "value": 0.2},
                {"dim": wrapper_kwargs["dim"], "value": 0.3},
                {"dim": wrapper_kwargs["dim"], "value": 0.4},
                {"dim": wrapper_kwargs["dim"], "value": 0.5},
                {"dim": wrapper_kwargs["dim"], "value": 0.6},
                {"dim": wrapper_kwargs["dim"], "value": 0.7},
                {"dim": wrapper_kwargs["dim"], "value": 0.8},
                {"dim": wrapper_kwargs["dim"], "value": 0.9},
            ]
        else:
            raise NotImplementedError()
        n_segments = 10
        wrapper_id = [wrapper_id] * 9
    elif exp_type == "continual_slowly":
        # linspace is upside-down without including endpoint and then flipped, to get right order
        # and actually not included starting point. Starting point is inserted afterwards.
        if wrapper_id == "OffsetAction":
            wrapper_kwargs = [
                {"dim": wrapper_kwargs["dim"], "value": v} for v in np.flip(np.linspace(-0.9, 0.0, 999, endpoint=False))
            ]
        elif wrapper_id == "ScaleAction":
            wrapper_kwargs = [
                {"dim": wrapper_kwargs["dim"], "value": v} for v in np.flip(np.linspace(0.1, 1.0, 999, endpoint=False))
            ]
        elif wrapper_id == "NoiseAction":
            wrapper_kwargs = [
                {"dim": wrapper_kwargs["dim"], "value": v} for v in np.flip(np.linspace(0.9, 0.0, 999, endpoint=False))
            ]
        else:
            raise NotImplementedError()
        n_segments = 500
        wrapper_id = [wrapper_id] * 499
    elif exp_type == "parallel_offset":
        if "value" in wrapper_kwargs:
            wrapper_kwargs = [
                {"dim": wrapper_kwargs["dim"], "value": wrapper_kwargs["value"]},
                {"dim": wrapper_kwargs["dim"], "value": 0.5},
            ]
        else:
            wrapper_kwargs = [
                {"dim": wrapper_kwargs["dim"]},
                {"dim": wrapper_kwargs["dim"], "value": 0.5},
            ]
        n_segments = 2
        wrapper_id = [wrapper_id, "OffsetAction"]
    elif exp_type == "parallel_scale":
        if "value" in wrapper_kwargs:
            wrapper_kwargs = [
                {"dim": wrapper_kwargs["dim"], "value": wrapper_kwargs["value"]},
                {"dim": wrapper_kwargs["dim"], "value": 0.5},
            ]
        else:
            wrapper_kwargs = [
                {"dim": wrapper_kwargs["dim"]},
                {"dim": wrapper_kwargs["dim"], "value": 0.5},
            ]
        n_segments = 2
        wrapper_id = [wrapper_id, "ScaleAction"]
    else:
        raise NotImplementedError()
    return wrapper_id, wrapper_kwargs, n_segments


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
    num_stack,
    num_contexts=None,
    wrapper_repeat=None,
):
    wrapper_kwargs = get_wrapper_kwargs(wrapper_id, wrapper_dim, wrapper_value, wrapper_repeat)
    wrapper_id, wrapper_kwargs, n_segments = get_experiment_wrapper_specs(
        exp_type,
        wrapper_id,
        wrapper_kwargs
    )
    if exp_type == "basic_robust":
        assert isinstance(wrapper_id, str)
        env, eval_envs = sequential_robust_experiment_setup(
            env_name=env_name,
            seed=seed,
            timesteps=timesteps,
            wrapper_id=wrapper_id,
            wrapper_kwargs=wrapper_kwargs,
            n_segments=n_segments,
            pixel=pixel,
            action_repeat=action_repeat,
            image_size=image_size,
            num_stack=num_stack,
            num_contexts=num_contexts
        )
    elif isinstance(wrapper_id, str):
        env, eval_envs = sequential_experiment_setup(
            env_name=env_name,
            seed=seed,
            timesteps=timesteps,
            wrapper_id=wrapper_id,
            wrapper_kwargs=wrapper_kwargs,
            n_segments=n_segments,
            pixel=pixel,
            action_repeat=action_repeat,
            image_size=image_size,
            num_stack=num_stack
        )
    elif isinstance(wrapper_id, list) and n_segments > 2:
        env, eval_envs = sequential_multiple_experiment_setup(
            env_name=env_name,
            seed=seed,
            timesteps=timesteps,
            wrapper_sequence=wrapper_id,
            wrapper_sequence_kwargs=wrapper_kwargs,
            pixel=pixel,
            action_repeat=action_repeat,
            image_size=image_size,
            num_stack=num_stack
        )
    elif isinstance(wrapper_id, list) and n_segments == 2:
        env, eval_envs = sequential_parallel_experiment_setup(
            env_name=env_name,
            seed=seed,
            timesteps=timesteps,
            wrapper_sequence=wrapper_id,
            wrapper_sequence_kwargs=wrapper_kwargs,
            n_segments=n_segments,
            pixel=pixel,
            action_repeat=action_repeat,
            image_size=image_size,
            num_stack=num_stack
        )
    return env, eval_envs
