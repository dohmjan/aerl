import gymnasium as gym

from stable_baselines3 import SAC, HerReplayBuffer
from aerl import InvertAction, ContextAwareObservation


def sac_gymnasium(timesteps=200000):
    env = gym.make("HalfCheetah-v4")
    # Invert the action effect (multiply by -1) in dimension 0 of the action.
    # For the multi-task setting, apply the action wrapper alternately every second episode.
    env = InvertAction(env, dim=0, toggle_at_episode=0.5)
    # Add task identifier to the observation.
    env = ContextAwareObservation(env)

    model = SAC("MlpPolicy", env, seed=2, verbose=1, stats_window_size=1)
    model.learn(total_timesteps=timesteps, log_interval=1)


def sac_fetch(timesteps=500000):
    env = gym.make("FetchPush-v2")
    env = InvertAction(env, dim=0, toggle_at_episode=0.5)
    env = DynamicsHintObservation(env)

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        seed=2,
        verbose=1,
        stats_window_size=1
    )
    model.learn(total_timesteps=timesteps, log_interval=40)


if __name__ == "__main__":
    sac_fetch()
