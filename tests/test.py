import gymnasium as gym
from gymnasium.wrappers.flatten_observation import FlattenObservation
from aerl import InvertAction, DynamicsHintObservation
import numpy as np


def test_sequential():
    env = gym.make("Hopper-v4")
    env = InvertAction(env, dim=0, toggle_at_step=2)
    env = InvertAction(env, dim=0, toggle_at_step=4)
    env = InvertAction(env, dim=1, toggle_at_step=6)
    states = []
    for i in range(10):
        _ = env.reset(seed=2)
        s, _, _, _, _ = env.step(np.array([1.0, -1.0, 0.0], dtype=np.float32))
        states.append(s)

    assert (states[0] == states[1]).all()
    assert (states[2] == states[3]).all()
    assert (states[2] != states[0]).all()
    assert (states[4] == states[0]).all()
    assert (states[6] == states[7]).all()
    assert (states[6] != states[0]).all()
    assert (states[6] != states[2]).all()


def test_scheduler():
    env = gym.make("Hopper-v4", terminate_when_unhealthy=False)
    env = InvertAction(env, dim=0, toggle_at_step=[1, 3, 4, 5])
    env = InvertAction(env, dim=0, toggle_at_step=6)
    env = InvertAction(env, dim=0, toggle_at_episode=[1, 2])
    states1 = []
    for i in range(8):
        _ = env.reset(seed=2)
        s, _, _, _, _ = env.step(np.array([1.0, -1.0, 0.0], dtype=np.float32))
        states1.append(s)
    assert (states1[0] != states1[1]).all()
    assert (states1[1] == states1[2]).all()
    assert (states1[2] != states1[3]).all()
    assert (states1[0] == states1[3]).all()
    assert (states1[4] != states1[5]).all()
    assert (states1[4] == states1[6]).all()
    assert (states1[6] == states1[7]).all()

    states2 = []
    _ = env.reset(seed=2)
    for i in range(2500):
        s, _, truncated, terminated, _ = env.step(np.array([1.0, -1.0, 0.0], dtype=np.float32))
        if truncated or terminated:
            _ = env.reset(seed=2)
        states2.append(s)
    assert (states2[0] != states2[1000]).all()
    assert (states2[0] == states2[2000]).all()
    assert (states2[0] == states1[1]).all()
    assert (states2[1000] == states1[0]).all()


def test_scheduler_step_freq():
    env = gym.make("Hopper-v4", terminate_when_unhealthy=False)
    env = InvertAction(env, dim=0, toggle_at_step=0.2)
    states = []
    for i in range(10):
        _ = env.reset(seed=2)
        s, _, _, _, _ = env.step(np.array([1.0, -1.0, 0.0], dtype=np.float32))
        states.append(s)
    assert (states[0] != states[1]).all()
    assert (states[1] == states[2]).all()
    assert (states[0] == states[5]).all()


def test_scheduler_episode_freq():
    env = gym.make("Hopper-v4", terminate_when_unhealthy=False)
    env = InvertAction(env, dim=0, toggle_at_episode=0.5)
    states = []
    _ = env.reset(seed=2)
    for i in range(3500):
        s, _, truncated, terminated, _ = env.step(np.array([1.0, -1.0, 0.0], dtype=np.float32))
        if truncated or terminated:
            _ = env.reset(seed=2)
        states.append(s)
    assert (states[0] != states[1000]).all()
    assert (states[0] == states[2000]).all()
    assert (states[1000] == states[3000]).all()


def test_hint():
    env = gym.make("Hopper-v4", terminate_when_unhealthy=False)
    env = FlattenObservation(env)
    env = InvertAction(env, dim=0, toggle_at_step=[2, 5, 10])
    env = FlattenObservation(env)
    env = InvertAction(env, dim=0, toggle_at_step=5)
    env = DynamicsHintObservation(env)
    states = []
    # TODO: Check this!
    for i in range(9):
        _ = env.reset(seed=2)
        s, _, _, _, _ = env.step(np.array([1.0, -1.0, 0.0], dtype=np.float32))
        states.append(s)

    assert states[0][-1] == 1
    assert states[0][-2] == 0
    assert states[0][-3] == 0
    assert states[0][-4] != 0 and states[0][-4] != 1

    assert states[1][-1] == 0
    assert states[1][-2] == 1
    assert states[1][-3] == 0

    assert states[3][-1] == 0
    assert states[3][-2] == 1
    assert states[3][-3] == 0

    assert states[4][-1] == 0
    assert states[4][-2] == 0
    assert states[4][-3] == 1

    try:
        _ = env.reset(seed=2)
        s, _, _, _, _ = env.step(np.array([1.0, -1.0, 0.0], dtype=np.float32))
        states.append(s)
        failed = False
    except ValueError:
        failed = True
    assert failed


def test_hint_episode_freq():
    env = gym.make("Hopper-v4", terminate_when_unhealthy=False)
    env = FlattenObservation(env)
    env = InvertAction(env, dim=0, toggle_at_episode=0.5)
    env = DynamicsHintObservation(env)
    states = []
    _ = env.reset(seed=2)
    for i in range(2500):
        s, _, truncated, terminated, _ = env.step(np.array([1.0, -1.0, 0.0], dtype=np.float32))
        if truncated or terminated:
            _ = env.reset(seed=2)
        states.append(s)
    assert states[0][-1] == 0
    assert states[0][-2] == 1
    assert states[1000][-1] == 1
    assert states[1000][-2] == 0
    assert states[2000][-1] == 0
    assert states[2000][-2] == 1


def test_mujoco():
    try:
        env = gym.make("Hopper-v4")
    except ImportError as e:
        raise ImportError(
            "Try installing `pip install gymnasium[mujoco]`."
        ) from e
    env = InvertAction(env, dim=0, toggle_at_step=[2, 5])
    env = DynamicsHintObservation(env)
    states = []
    for i in range(10):
        _ = env.reset(seed=2)
        s, _, _, _, _ = env.step(np.ones(env.action_space.shape, dtype=env.action_space.dtype))
        states.append(s)


def test_dmc():
    try:
        env = gym.make("dm_control/walker-walk-v0")
    except ImportError as e:
        raise ImportError(
            "Try installing `pip install shimmy[dm-control]`."
        ) from e
    env = FlattenObservation(env)
    env = InvertAction(env, dim=0, toggle_at_step=[2, 5])
    env = DynamicsHintObservation(env)
    states = []
    for i in range(10):
        _ = env.reset(seed=2)
        s, _, _, _, _ = env.step(np.ones(env.action_space.shape, dtype=env.action_space.dtype))
        states.append(s)


def test_fetch():
    try:
        env = gym.make("FetchReach-v2")
    except ImportError as e:
        raise ImportError(
            "Try installing `pip install gymnasium-robotics`."
        ) from e
    env = InvertAction(env, dim=0, toggle_at_step=[2, 5])
    env = DynamicsHintObservation(env)
    states = []
    for i in range(10):
        _ = env.reset(seed=2)
        s, _, _, _, _ = env.step(np.ones(env.action_space.shape, dtype=env.action_space.dtype))
        states.append(s)


if __name__ == "__main__":
    test_sequential()
    test_scheduler()
    test_scheduler_step_freq()
    test_scheduler_episode_freq()
    test_hint()
    test_hint_episode_freq()
    test_mujoco()
    test_dmc()
    test_fetch()
