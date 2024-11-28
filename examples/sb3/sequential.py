import gymnasium as gym
from gymnasium.wrappers.flatten_observation import FlattenObservation

from stable_baselines3 import SAC, PPO, HerReplayBuffer
from aerl import InvertAction


def sac_gymnasium(timesteps=200000):
    env = gym.make("HalfCheetah-v4")
    # Invert the action effect (multiply by -1) in dimension 0 of the action.
    # For the sequential setting, apply the action wrapper add a certain time during training.
    # Here, the wrapper is activated from half of the maximum time steps.
    env = InvertAction(env, dim=0, toggle_at_step=timesteps // 2)

    model = SAC("MlpPolicy", env, seed=2, verbose=1, stats_window_size=1)
    model.learn(total_timesteps=timesteps)


def sac_dmc(timesteps=200000):
    env = gym.make("dm_control/walker-walk-v0")
    if "dm_control" in env.spec.id:
        env = FlattenObservation(env)
    env = InvertAction(env, dim=0, toggle_at_step=timesteps // 2)

    model = SAC("MlpPolicy", env, seed=2, verbose=1, stats_window_size=1)
    model.learn(total_timesteps=timesteps)


def ppo_dmc(timesteps=200000):
    env = gym.make("dm_control/walker-walk-v0")
    if "dm_control" in env.spec.id:
        env = FlattenObservation(env)
    env = InvertAction(env, dim=0, toggle_at_step=timesteps // 2)

    model = PPO("MlpPolicy", env, seed=2, verbose=1, stats_window_size=1)
    model.learn(total_timesteps=timesteps)


def sac_fetch(timesteps=200000):
    env = gym.make("FetchPush-v2")
    env = InvertAction(env, dim=0, toggle_at_step=timesteps // 2)

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        seed=2,
        verbose=1,
        stats_window_size=1
    )
    model.learn(total_timesteps=timesteps)


if __name__ == "__main__":
    sac_dmc()
